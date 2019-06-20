fastai.callbacks.one_cycle.OneCycleScheduler
fastai.callback.annealing_cos
fastai.callback.Scheduler
fastai.callback.CallbackHandler
fastai.fastprogress.fastprogress.NBProgressBar
fastai.data_block.ItemList
fastai.basic_data.DataBunch
fastai.vision.gan.AdaptiveLoss
fastai.callback.Callback
fastai.basic_train.LearnerCallback
fastai.torch_core.get_model

OneCycleScheduler
	LearnerCallback
		Callback

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


AdamW = partial(torch.optim.adam.Adam, betas=(0.9,0.99))



fit_one_cycle
	callbacks = listify(callbacks)
    callbacks.append(OneCycleScheduler)
    learn.fit()
    	fit()


def fit(epochs:int, learn:BasicLearner, callbacks:Optional[CallbackList]=None, metrics:OptMetrics=None)->None:
    "Fit the `model` on `data` and learn using `loss_func` and `opt`."
    assert len(learn.data.train_dl) != 0, f"""Your training dataloader is empty, can't train a model.
        Use a smaller batch size (batch size={learn.data.train_dl.batch_size} for {len(learn.data.train_dl.dataset)} elements)."""
    cb_handler = CallbackHandler(callbacks, metrics)
    pbar = master_bar(range(epochs))
    cb_handler.on_train_begin(epochs, pbar=pbar, metrics=metrics)

    exception=False
    try:
        for epoch in pbar:
            learn.model.train()
            cb_handler.set_dl(learn.data.train_dl)
            cb_handler.on_epoch_begin()
            for xb,yb in progress_bar(learn.data.train_dl, parent=pbar):
                xb, yb = cb_handler.on_batch_begin(xb, yb)
                loss = loss_batch(learn.model, xb, yb, learn.loss_func, learn.opt, cb_handler)
                if cb_handler.on_batch_end(loss): break

            if not cb_handler.skip_validate and not learn.data.empty_val:
                val_loss = validate(learn.model, learn.data.valid_dl, loss_func=learn.loss_func,
                                       cb_handler=cb_handler, pbar=pbar)
            else: val_loss=None
            if cb_handler.on_epoch_end(val_loss): break
    except Exception as e:
        exception = e
        raise
    finally: cb_handler.on_train_end(exception)