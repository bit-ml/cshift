class BasicExpert():
    TASK_CLASSIFICATION = 0
    TASK_REGRESSION = 1

    def get_task_type(self):
        return BasicExpert.TASK_REGRESSION

    def no_maps_as_nn_input(self):
        return self.n_maps

    def no_maps_as_nn_output(self):
        return self.n_maps

    def no_maps_as_ens_input(self):
        return self.n_maps

    def postprocess_eval(self, nn_outp):
        '''
        POST PROCESSING eval - posprocess operations for evaluation (e.g. scale/normalize)
        '''
        return nn_outp.clamp(min=0, max=1)

    def postprocess_ensemble_eval(self, nn_outp):
        '''
        POST PROCESS ENSEMBLE result
        '''
        return nn_outp.clamp(min=0, max=1)

    def gt_train_transform(self, x):
        '''
        GT train - added for normals expert only
        '''
        return x

    def gt_eval_transform(self, x, n_classes):
        '''
        GT eval - added for sem segm expert only
        '''
        return x

    def exp_eval_transform(self, x, n_classes):
        '''
        EXP eval - added for sem segm expert only
        '''
        return x

    def gt_to_inp_transform(self, x, n_classes):
        '''
        GT ensemble eval - added for sem segm expert only
        '''
        return x

    def test_gt(self, loss_fct, pred, target):
        '''
        GT eval - added for depth GT only (mask wrong pixels in GT)
                - used for testing purpose, to treat different nan values of the gt 
        '''
        return loss_fct(pred, target)
