python paper_exp/train_abf_nn.py exp=abf
python paper_exp/obf_basic.py operation=bus14_continuous exp=obf_basic
python paper_exp/obf_sco.py operation=bus14_continuous exp=obf_sco
python paper_exp/obf_uncer.py operation=bus14_continuous exp=obf_uncer exp.budget_ratio=0.01 exp.train_config.M_DP=1e3 exp.train_config.M_RD=1e3
python paper_exp/obf_uncer.py operation=bus14_continuous exp=obf_uncer exp.budget_ratio=0.03 exp.train_config.M_DP=1e3 exp.train_config.M_RD=1e3
python paper_exp/obf_uncer.py operation=bus14_continuous exp=obf_uncer exp.budget_ratio=0.05 exp.train_config.M_DP=1e3 exp.train_config.M_RD=1e3
python paper_exp/obf_uncer.py operation=bus14_continuous exp=obf_uncer exp.budget_ratio=0.07 exp.train_config.M_DP=1e3 exp.train_config.M_RD=1e4
python paper_exp/obf_sco_grad.py operation=bus14_continuous exp=obf_sco