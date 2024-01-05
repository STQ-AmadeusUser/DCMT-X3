import logging
from .derive_arch import BaseArchGenerate


class ArchGenerate(BaseArchGenerate):
    def __init__(self, super_network, config):
        super(ArchGenerate, self).__init__(super_network, config)
    
    def derive_archs(self, betas, head_alphas, stack_alphas, if_display=True):
        
        self.update_arch_params(betas, head_alphas, stack_alphas)

        # [[ch, head_op, [stack_op], num_layers, stride], ..., [...]]
        derived_archs = [
            [[3, self.config.MODEL.NECK.IN_CHANNEL], self.config.MODEL.BACKBONE.NAME],
            [[self.config.MODEL.NECK.IN_CHANNEL, self.config.MODEL.NECK.OUT_CHANNEL], self.config.MODEL.NECK.NAME],
            [[self.config.MODEL.PRE_FUSION.IN_CHANNEL, self.config.MODEL.PRE_FUSION.HID_CHANNEL, self.config.MODEL.PRE_FUSION.OUT_CHANNEL],
             self.config.MODEL.PRE_FUSION.STRIDE, self.config.MODEL.PRE_FUSION.NAME],
        ]
        ch_path, derived_chs = self.derive_chs()

        layer_count = 0
        for i, (ch_idx, ch) in enumerate(zip(ch_path, derived_chs)):
            if ch_idx == 0 or i == len(derived_chs)-1:
                continue

            block_idx = ch_idx - 1
            input_config = self.input_configs[block_idx]

            head_id = input_config['in_block_idx'].index(ch_path[i-1])
            head_alpha = self.head_alphas[block_idx][head_id]
            head_op = self.derive_ops(head_alpha, 'head')

            stride = input_config['strides'][input_config['in_block_idx'].index(ch_path[i-1])]

            stack_ops = []
            for stack_alpha in self.stack_alphas[block_idx]:
                stack_op = self.derive_ops(stack_alpha, 'stack')
                if stack_op != 'skip_connect':
                    stack_ops.append(stack_op)
                    layer_count += 1

            derived_archs.append(
                [[derived_chs[i-1], ch], head_op, stack_ops, len(stack_ops), stride]
            )
        derived_archs.append([[derived_chs[-2], self.config.SEARCH.LAST_DIM], 'collector'])
        derived_archs.append([[self.config.SEARCH.LAST_DIM, self.config.SEARCH.LAST_DIM], 'prediction'])

        layer_count += 5  # backbone, neck, match, collector, prediction
        if if_display:
            logging.info('Derived arch: \n' + '|\n'.join(map(str, derived_archs)))
            logging.info('Total {} layers.'.format(layer_count))

        return derived_archs
