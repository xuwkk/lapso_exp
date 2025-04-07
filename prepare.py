import hydra
from omegaconf import DictConfig
from pso import prepare_grid_from_pypower, prepare_data, get_dataset_np, UC_DISCRETE, prepare_pf_limit

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    grid_xlsx = prepare_grid_from_pypower(cfg.grid, force_new=cfg.force_new_grid)
    prepare_data(grid_xlsx, cfg.grid.data_dir, cfg.random_seed, cfg.force_new_data)
    
    feature_all, load_all, solar_all, wind_all = get_dataset_np(cfg.grid)
    
    assert cfg.operation.with_binary, "It is suggested to rescale the power flow limit with discrete UC"
    
    uc = UC_DISCRETE(grid_xlsx=grid_xlsx, operation_cfg=cfg.operation)
    
    prepare_pf_limit(load_all, solar_all, wind_all, cfg.grid, cfg.optimization, uc)
    
if __name__ == "__main__":
    main()