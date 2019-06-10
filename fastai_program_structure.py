fastai.callbacks.one_cycle.OneCycleScheduler
fastai.callback.CallbackHandler
fastai.fastprogress.fastprogress.NBProgressBar
fastai.data_block.ItemList
fastai.basic_data.DataBunch
fastai.vision.gan.AdaptiveLoss

fasterai.loss.FeatureLoss

torch.utils.data.dataloader.DataLoader

ImageImageList
	ImageList 
		ItemList

LabelList
	torch.utils.data.dataset.Dataset

LabelLists
	ItemLists

learner.fit_one_cycle()
	callbacks.append(OneCycleScheduler())
	learner.fit(cyc_len, max_lr, wd=wd, callbacks=callbacks)
		cb_handler = CallbackHandler(callbacks, metrics)
		cb_handler.on_train_begin(epochs, pbar=pbar, metrics=metrics)
			cb_handler.__call__('train_begin', metrics_names=names)
				cb_handler._call_and_update(cb, cb_name, **kwargs)
		learn.model.train()
        cb_handler.set_dl(learn.data.train_dl)
        cb_handler.on_epoch_begin()

        for xb,yb in progress_bar(learn.data.train_dl, parent=pbar): # fastai.fastprogress.fastprogress.NBProgressBar
	        xb, yb = cb_handler.on_batch_begin(xb, yb)
	        loss = loss_batch(learn.model, xb, yb, learn.loss_func, learn.opt, cb_handler)
	        if cb_handler.on_batch_end(loss): break

        cb_handler.on_train_end(exception)


get_colorize_data()
	ImageImageList.from_folder(cls, path) # classmethod
		cls(fastai.data_block.get_files(path, extensions, recurse=recurse, include=include, presort=presort), path=path)
	ItemList.use_partial_data()
	fastai.data_block.ItemLists.split_by_rand_pct(0.1, seed=random_seed)
	fastai.data_block.ItemLists.label_from_func() # x, y is assigned in _label_from_list(), convert LableList to LabelLists in ItemLists's constructor

LabelLists.transform() # ItemLists.transform()
	LabelList.transform() # register transformations
		_check_kwargs() # do transform parameter check



databunch()
	ItemList # not implemented
	ItemLists # not implemented
	LabelLists # implemented