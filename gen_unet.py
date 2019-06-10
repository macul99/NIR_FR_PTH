fasterai.generator.gen_learner_wide()
	fasterai.generator.unet_learner_wide()
		fastai.vision.learner.cnn_config()
		fastai.vision.learner.create_body()
		fasterai.unet.DynamicUnetWide()
			fastai.vision.NormType
			fastai.callbacks.hooks.model_sizes()
			fastai.callbacks.hooks.hook_outputs()
		
		fastai.basic_train.Learner()
		fastai.basic_train.Learner().split()
		fastai.basic_train.Learner().freeze() # if pretrained
		fastai.torch_core.apply_init()


fastai.basic_train.Learner().fit_one_cycle()

