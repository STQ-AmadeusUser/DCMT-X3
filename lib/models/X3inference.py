from lib.utils.deploy_helper import print_properties


class ModelDeploy(object):
    def __init__(self, configs, models):
        super(ModelDeploy, self).__init__()
        self.template_size = configs.TRAIN.TEMPLATE_SIZE
        self.search_size = configs.TRAIN.SEARCH_SIZE
        self.score_size = configs.TRAIN.SCORE_SIZE
        self.init_arch(models)

    def init_arch(self, model):
        self.inference = model['inference']

    def template(self, z, z_bbox):
        self.z = z
        self.z_bbox = z_bbox

    def track(self, x):
        cls, reg = self.inference[0].forward([self.z_bbox, self.z, x])
        # print('cls feature map:')
        # print_properties(cls.properties)
        # print('reg feature map:')
        # print_properties(reg.properties)
        return cls.buffer, reg.buffer
