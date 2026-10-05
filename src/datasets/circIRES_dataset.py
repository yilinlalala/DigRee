class circIRESinfos:
    """Dataset statistics stored inside each checkpoint (hyper_parameters['dataset_infos']).

    The checkpoints pickle this class by its import path
    (datasets.circIRES_dataset.circIRESinfos), so the module path and class name
    must stay unchanged for torch.load to restore it. All attributes come from
    the checkpoint itself.
    """
