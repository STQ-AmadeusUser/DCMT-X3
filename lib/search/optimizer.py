import torch
import torch.nn as nn
import utils.model_helper as loader
import utils.lr_scheduler as learner
from models.dropped_model import Dropped_Network


class Optimizer(object):

    def __init__(self, model, cls_criterion, reg_criterion, config, tracking_config):
        self.config = config
        self.tracking_config = tracking_config
        self.weight_sample_num = self.config.search_params.weight_sample_num
        self.cls_loss = cls_criterion
        self.reg_loss = reg_criterion
        self.Dropped_Network = lambda model: Dropped_Network(model, softmax_temp=config.search_params.softmax_temp)

        self.build_weight_opt_lr(model)
        self.arch_optimizer = torch.optim.Adam(
            [{'params': model.module.arch_alpha_params, 'lr': config.optim.arch.alpha_lr},
             {'params': model.module.arch_beta_params, 'lr': config.optim.arch.beta_lr}],
            betas=(0.5, 0.999),
            weight_decay=config.optim.arch.weight_decay)

    def arch_step(self, input_valid, target_valid, model, search_stage):
        head_sampled_w_old, alpha_head_index = model.module.sample_branch('head', 2, search_stage=search_stage)
        stack_sampled_w_old, alpha_stack_index = model.module.sample_branch('stack', 2, search_stage=search_stage)
        self.arch_optimizer.zero_grad()

        dropped_model = nn.DataParallel(self.Dropped_Network(model))
        logits, sub_obj = dropped_model(input_valid)
        sub_obj = torch.mean(sub_obj)
        loss = self.criterion(logits, target_valid)
        if self.config.optim.if_sub_obj:
            loss_sub_obj = torch.log(sub_obj) / torch.log(torch.tensor(self.config.optim.sub_obj.log_base))
            sub_loss_factor = self.config.optim.sub_obj.sub_loss_factor
            loss += loss_sub_obj * sub_loss_factor
        loss.backward()
        self.arch_optimizer.step()

        self.rescale_arch_params(head_sampled_w_old,
                                 stack_sampled_w_old,
                                 alpha_head_index,
                                 alpha_stack_index,
                                 model)
        return logits.detach(), loss.item(), sub_obj.item()

    def weight_step(self, *args, **kwargs):
        return self.weight_step_(*args, **kwargs)

    def weight_step_(self, inputs, model, search_stage):
        _, _ = model.module.sample_branch('head', self.weight_sample_num, search_stage=search_stage)
        _, _ = model.module.sample_branch('stack', self.weight_sample_num, search_stage=search_stage)

        dropped_model = nn.DataParallel(self.Dropped_Network(model))
        z, x, bbox = inputs['template'], inputs['search'], inputs['template_bbox']
        cls_pred, reg_pred, sub_obj = dropped_model(z, x, bbox)
        sub_obj = torch.mean(sub_obj)
        cls_label, reg_label, reg_weight = inputs['cls_label'], inputs['reg_label'], inputs['reg_weight']
        cls_loss = self.cls_loss(cls_pred, cls_label)
        reg_loss, iou = self.reg_loss(reg_pred, reg_label, reg_weight)
        cls_loss = torch.mean(cls_loss)
        reg_loss = torch.mean(reg_loss)
        iou = torch.mean(iou)
        loss = self.tracking_config.TRAIN.CLS_WEIGHT * cls_loss + self.tracking_config.TRAIN.REG_WEIGHT * reg_loss
        loss = torch.mean(loss)

        self.weight_optimizer.zero_grad()
        loss.backward()

        if self.tracking_config.TRAIN.CLIP_GRAD:
            torch.nn.utils.clip_grad_norm(model.parameters(), 10)  # gradient clip

        if loader.is_valid_number(loss.item()):
            self.weight_optimizer.step()

        return iou.detach().item(), loss.item(), sub_obj.item()

    def valid_step(self, input_valid, target_valid, model):
        _, _ = model.module.sample_branch('head', 1, training=False)
        _, _ = model.module.sample_branch('stack', 1, training=False)

        dropped_model = nn.DataParallel(self.Dropped_Network(model))
        logits, sub_obj = dropped_model(input_valid)
        sub_obj = torch.mean(sub_obj)
        loss = self.criterion(logits, target_valid)

        return logits, loss.item(), sub_obj.item()

    def rescale_arch_params(self, alpha_head_weights_drop, 
                            alpha_stack_weights_drop,
                            alpha_head_index,
                            alpha_stack_index,
                            model):

        def comp_rescale_value(old_weights, new_weights, index):
            old_exp_sum = old_weights.exp().sum()
            new_drop_arch_params = torch.gather(new_weights, dim=-1, index=index)
            new_exp_sum = new_drop_arch_params.exp().sum()
            rescale_value = torch.log(old_exp_sum / new_exp_sum)
            rescale_mat = torch.zeros_like(new_weights).scatter_(0, index, rescale_value)
            return rescale_value, rescale_mat
        
        def rescale_params(old_weights, new_weights, indices):
            for i, (old_weights_block, indices_block) in enumerate(zip(old_weights, indices)):
                for j, (old_weights_branch, indices_branch) in enumerate(zip(old_weights_block, indices_block)):
                    rescale_value, rescale_mat = comp_rescale_value(old_weights_branch,
                                                                    new_weights[i][j],
                                                                    indices_branch)
                    new_weights[i][j].data.add_(rescale_mat)

        # rescale the arch params for head layers
        rescale_params(alpha_head_weights_drop, model.module.alpha_head_weights, alpha_head_index)
        # rescale the arch params for stack layers
        rescale_params(alpha_stack_weights_drop, model.module.alpha_stack_weights, alpha_stack_index)

    def set_param_grad_state(self, stage):
        def set_grad_state(params, state):
            for group in params:
                for param in group['params']:
                    param.requires_grad_(state)
        if stage == 'Arch':
            state_list = [True, False] # [arch, weight]
        elif stage == 'Weights':
            state_list = [False, True]
        else:
            state_list = [False, False]
        set_grad_state(self.arch_optimizer.param_groups, state_list[0])
        set_grad_state(self.weight_optimizer.param_groups, state_list[1])

    def build_weight_opt_lr(self, model, epoch=None):
        if epoch is None:
            weight_optimizer, lr_scheduler = learner.build_siamese_opt_lr(self.tracking_config, model,
                                                                          self.tracking_config.TRAIN.START_EPOCH)
        else:
            weight_optimizer, lr_scheduler = learner.build_siamese_opt_lr(self.tracking_config, model, epoch)
        self.weight_optimizer = weight_optimizer
        self.scheduler = lr_scheduler
