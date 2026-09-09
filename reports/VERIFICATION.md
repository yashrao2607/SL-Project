# Verification report

Generated 2026-09-09 by `scripts/verify_report.py`.

Checks run: 542; failed: 0.

| Check | Result | Detail |
|---|---|---|
| tier1_run_count | PASS | 2800 rows, expected 2800 |
| tier1_no_nan_primary | PASS | NaN primary metric present |
| every_cell_has_25_runs | PASS | min 25 max 25 |
| seeds_are_charter_seeds | PASS | [42, 123, 456, 789, 1011] |
| folds_0_to_4 | PASS |  |
| identical_partitions_across_models | PASS |  |
| random_gcn_bbbp_roc_auc_mean | PASS | recomputed 0.8987 vs report 0.899 |
| diff_random_gcn_bbbp | PASS | recomputed 0.0070 vs 0.007 |
| wilcoxon_p_random_gcn_bbbp | PASS | recomputed p=0.3810 vs 0.381 |
| npairs_random_gcn_bbbp | PASS | 25 pairs |
| random_gcn_esol_rmse_mean | PASS | recomputed 0.4571 vs report 0.457 |
| diff_random_gcn_esol | PASS | recomputed 0.0717 vs 0.072 |
| wilcoxon_p_random_gcn_esol | PASS | recomputed p=0.0000 vs 0.0 |
| npairs_random_gcn_esol | PASS | 25 pairs |
| random_gcn_freesolv_rmse_mean | PASS | recomputed 0.3383 vs report 0.338 |
| diff_random_gcn_freesolv | PASS | recomputed -0.0303 vs -0.03 |
| wilcoxon_p_random_gcn_freesolv | PASS | recomputed p=0.0451 vs 0.0451 |
| npairs_random_gcn_freesolv | PASS | 25 pairs |
| random_gcn_estrogen-alpha_roc_auc_mean | PASS | recomputed 0.9560 vs report 0.956 |
| diff_random_gcn_estrogen-alpha | PASS | recomputed 0.0029 vs 0.003 |
| wilcoxon_p_random_gcn_estrogen-alpha | PASS | recomputed p=0.0714 vs 0.0714 |
| npairs_random_gcn_estrogen-alpha | PASS | 25 pairs |
| random_gcn_estrogen-beta_roc_auc_mean | PASS | recomputed 0.9125 vs report 0.913 |
| diff_random_gcn_estrogen-beta | PASS | recomputed 0.0196 vs 0.02 |
| wilcoxon_p_random_gcn_estrogen-beta | PASS | recomputed p=0.0000 vs 0.0 |
| npairs_random_gcn_estrogen-beta | PASS | 25 pairs |
| random_gcn_metstab-high_roc_auc_mean | PASS | recomputed 0.8476 vs report 0.848 |
| diff_random_gcn_metstab-high | PASS | recomputed 0.0566 vs 0.057 |
| wilcoxon_p_random_gcn_metstab-high | PASS | recomputed p=0.0001 vs 0.0001 |
| npairs_random_gcn_metstab-high | PASS | 25 pairs |
| random_gcn_metstab-low_roc_auc_mean | PASS | recomputed 0.8471 vs report 0.847 |
| diff_random_gcn_metstab-low | PASS | recomputed 0.0824 vs 0.082 |
| wilcoxon_p_random_gcn_metstab-low | PASS | recomputed p=0.0000 vs 0.0 |
| npairs_random_gcn_metstab-low | PASS | 25 pairs |
| random_hybrid_bbbp_roc_auc_mean | PASS | recomputed 0.9110 vs report 0.911 |
| diff_random_hybrid_bbbp | PASS | recomputed 0.0192 vs 0.019 |
| wilcoxon_p_random_hybrid_bbbp | PASS | recomputed p=0.0012 vs 0.0012 |
| npairs_random_hybrid_bbbp | PASS | 25 pairs |
| random_hybrid_esol_rmse_mean | PASS | recomputed 0.4913 vs report 0.491 |
| diff_random_hybrid_esol | PASS | recomputed 0.1058 vs 0.106 |
| wilcoxon_p_random_hybrid_esol | PASS | recomputed p=0.0000 vs 0.0 |
| npairs_random_hybrid_esol | PASS | 25 pairs |
| random_hybrid_freesolv_rmse_mean | PASS | recomputed 0.3786 vs report 0.379 |
| diff_random_hybrid_freesolv | PASS | recomputed 0.0101 vs 0.01 |
| wilcoxon_p_random_hybrid_freesolv | PASS | recomputed p=0.2200 vs 0.22 |
| npairs_random_hybrid_freesolv | PASS | 25 pairs |
| random_hybrid_estrogen-alpha_roc_auc_mean | PASS | recomputed 0.9687 vs report 0.969 |
| diff_random_hybrid_estrogen-alpha | PASS | recomputed 0.0156 vs 0.016 |
| wilcoxon_p_random_hybrid_estrogen-alpha | PASS | recomputed p=0.0000 vs 0.0 |
| npairs_random_hybrid_estrogen-alpha | PASS | 25 pairs |
| random_hybrid_estrogen-beta_roc_auc_mean | PASS | recomputed 0.9220 vs report 0.922 |
| diff_random_hybrid_estrogen-beta | PASS | recomputed 0.0291 vs 0.029 |
| wilcoxon_p_random_hybrid_estrogen-beta | PASS | recomputed p=0.0000 vs 0.0 |
| npairs_random_hybrid_estrogen-beta | PASS | 25 pairs |
| random_hybrid_metstab-high_roc_auc_mean | PASS | recomputed 0.8983 vs report 0.898 |
| diff_random_hybrid_metstab-high | PASS | recomputed 0.1073 vs 0.107 |
| wilcoxon_p_random_hybrid_metstab-high | PASS | recomputed p=0.0000 vs 0.0 |
| npairs_random_hybrid_metstab-high | PASS | 25 pairs |
| random_hybrid_metstab-low_roc_auc_mean | PASS | recomputed 0.8635 vs report 0.863 |
| diff_random_hybrid_metstab-low | PASS | recomputed 0.0988 vs 0.099 |
| wilcoxon_p_random_hybrid_metstab-low | PASS | recomputed p=0.0000 vs 0.0 |
| npairs_random_hybrid_metstab-low | PASS | 25 pairs |
| random_mat_bbbp_roc_auc_mean | PASS | recomputed 0.8917 vs report 0.892 |
| random_mat_esol_rmse_mean | PASS | recomputed 0.3854 vs report 0.385 |
| random_mat_freesolv_rmse_mean | PASS | recomputed 0.3686 vs report 0.369 |
| random_mat_estrogen-alpha_roc_auc_mean | PASS | recomputed 0.9531 vs report 0.953 |
| random_mat_estrogen-beta_roc_auc_mean | PASS | recomputed 0.8929 vs report 0.893 |
| random_mat_metstab-high_roc_auc_mean | PASS | recomputed 0.7910 vs report 0.791 |
| random_mat_metstab-low_roc_auc_mean | PASS | recomputed 0.7647 vs report 0.765 |
| random_mat_noattention_bbbp_roc_auc_mean | PASS | recomputed 0.8551 vs report 0.855 |
| diff_random_mat_noattention_bbbp | PASS | recomputed -0.0366 vs -0.037 |
| wilcoxon_p_random_mat_noattention_bbbp | PASS | recomputed p=0.0000 vs 0.0 |
| npairs_random_mat_noattention_bbbp | PASS | 25 pairs |
| random_mat_noattention_esol_rmse_mean | PASS | recomputed 0.4530 vs report 0.453 |
| diff_random_mat_noattention_esol | PASS | recomputed 0.0675 vs 0.068 |
| wilcoxon_p_random_mat_noattention_esol | PASS | recomputed p=0.0000 vs 0.0 |
| npairs_random_mat_noattention_esol | PASS | 25 pairs |
| random_mat_noattention_freesolv_rmse_mean | PASS | recomputed 0.4577 vs report 0.458 |
| diff_random_mat_noattention_freesolv | PASS | recomputed 0.0891 vs 0.089 |
| wilcoxon_p_random_mat_noattention_freesolv | PASS | recomputed p=0.0000 vs 0.0 |
| npairs_random_mat_noattention_freesolv | PASS | 25 pairs |
| random_mat_noattention_estrogen-alpha_roc_auc_mean | PASS | recomputed 0.9251 vs report 0.925 |
| diff_random_mat_noattention_estrogen-alpha | PASS | recomputed -0.0280 vs -0.028 |
| wilcoxon_p_random_mat_noattention_estrogen-alpha | PASS | recomputed p=0.0000 vs 0.0 |
| npairs_random_mat_noattention_estrogen-alpha | PASS | 25 pairs |
| random_mat_noattention_estrogen-beta_roc_auc_mean | PASS | recomputed 0.8523 vs report 0.852 |
| diff_random_mat_noattention_estrogen-beta | PASS | recomputed -0.0406 vs -0.041 |
| wilcoxon_p_random_mat_noattention_estrogen-beta | PASS | recomputed p=0.0000 vs 0.0 |
| npairs_random_mat_noattention_estrogen-beta | PASS | 25 pairs |
| random_mat_noattention_metstab-high_roc_auc_mean | PASS | recomputed 0.7335 vs report 0.734 |
| diff_random_mat_noattention_metstab-high | PASS | recomputed -0.0575 vs -0.057 |
| wilcoxon_p_random_mat_noattention_metstab-high | PASS | recomputed p=0.0003 vs 0.0003 |
| npairs_random_mat_noattention_metstab-high | PASS | 25 pairs |
| random_mat_noattention_metstab-low_roc_auc_mean | PASS | recomputed 0.6446 vs report 0.645 |
| diff_random_mat_noattention_metstab-low | PASS | recomputed -0.1201 vs -0.12 |
| wilcoxon_p_random_mat_noattention_metstab-low | PASS | recomputed p=0.0000 vs 0.0 |
| npairs_random_mat_noattention_metstab-low | PASS | 25 pairs |
| random_mat_nodistance_bbbp_roc_auc_mean | PASS | recomputed 0.8866 vs report 0.887 |
| diff_random_mat_nodistance_bbbp | PASS | recomputed -0.0051 vs -0.005 |
| wilcoxon_p_random_mat_nodistance_bbbp | PASS | recomputed p=0.4751 vs 0.4751 |
| npairs_random_mat_nodistance_bbbp | PASS | 25 pairs |
| random_mat_nodistance_esol_rmse_mean | PASS | recomputed 0.3909 vs report 0.391 |
| diff_random_mat_nodistance_esol | PASS | recomputed 0.0055 vs 0.006 |
| wilcoxon_p_random_mat_nodistance_esol | PASS | recomputed p=0.6528 vs 0.6528 |
| npairs_random_mat_nodistance_esol | PASS | 25 pairs |
| random_mat_nodistance_freesolv_rmse_mean | PASS | recomputed 0.3799 vs report 0.38 |
| diff_random_mat_nodistance_freesolv | PASS | recomputed 0.0113 vs 0.011 |
| wilcoxon_p_random_mat_nodistance_freesolv | PASS | recomputed p=0.4578 vs 0.4578 |
| npairs_random_mat_nodistance_freesolv | PASS | 25 pairs |
| random_mat_nodistance_estrogen-alpha_roc_auc_mean | PASS | recomputed 0.9516 vs report 0.952 |
| diff_random_mat_nodistance_estrogen-alpha | PASS | recomputed -0.0014 vs -0.001 |
| wilcoxon_p_random_mat_nodistance_estrogen-alpha | PASS | recomputed p=0.2099 vs 0.2099 |
| npairs_random_mat_nodistance_estrogen-alpha | PASS | 25 pairs |
| random_mat_nodistance_estrogen-beta_roc_auc_mean | PASS | recomputed 0.8883 vs report 0.888 |
| diff_random_mat_nodistance_estrogen-beta | PASS | recomputed -0.0046 vs -0.005 |
| wilcoxon_p_random_mat_nodistance_estrogen-beta | PASS | recomputed p=0.0903 vs 0.0903 |
| npairs_random_mat_nodistance_estrogen-beta | PASS | 25 pairs |
| random_mat_nodistance_metstab-high_roc_auc_mean | PASS | recomputed 0.7888 vs report 0.789 |
| diff_random_mat_nodistance_metstab-high | PASS | recomputed -0.0022 vs -0.002 |
| wilcoxon_p_random_mat_nodistance_metstab-high | PASS | recomputed p=0.2996 vs 0.2996 |
| npairs_random_mat_nodistance_metstab-high | PASS | 25 pairs |
| random_mat_nodistance_metstab-low_roc_auc_mean | PASS | recomputed 0.7715 vs report 0.771 |
| diff_random_mat_nodistance_metstab-low | PASS | recomputed 0.0067 vs 0.007 |
| wilcoxon_p_random_mat_nodistance_metstab-low | PASS | recomputed p=0.2099 vs 0.2099 |
| npairs_random_mat_nodistance_metstab-low | PASS | 25 pairs |
| random_mat_nograph_bbbp_roc_auc_mean | PASS | recomputed 0.8826 vs report 0.883 |
| diff_random_mat_nograph_bbbp | PASS | recomputed -0.0092 vs -0.009 |
| wilcoxon_p_random_mat_nograph_bbbp | PASS | recomputed p=0.0275 vs 0.0275 |
| npairs_random_mat_nograph_bbbp | PASS | 25 pairs |
| random_mat_nograph_esol_rmse_mean | PASS | recomputed 0.4108 vs report 0.411 |
| diff_random_mat_nograph_esol | PASS | recomputed 0.0254 vs 0.025 |
| wilcoxon_p_random_mat_nograph_esol | PASS | recomputed p=0.0903 vs 0.0903 |
| npairs_random_mat_nograph_esol | PASS | 25 pairs |
| random_mat_nograph_freesolv_rmse_mean | PASS | recomputed 0.3760 vs report 0.376 |
| diff_random_mat_nograph_freesolv | PASS | recomputed 0.0074 vs 0.007 |
| wilcoxon_p_random_mat_nograph_freesolv | PASS | recomputed p=0.8119 vs 0.8119 |
| npairs_random_mat_nograph_freesolv | PASS | 25 pairs |
| random_mat_nograph_estrogen-alpha_roc_auc_mean | PASS | recomputed 0.9470 vs report 0.947 |
| diff_random_mat_nograph_estrogen-alpha | PASS | recomputed -0.0060 vs -0.006 |
| wilcoxon_p_random_mat_nograph_estrogen-alpha | PASS | recomputed p=0.0000 vs 0.0 |
| npairs_random_mat_nograph_estrogen-alpha | PASS | 25 pairs |
| random_mat_nograph_estrogen-beta_roc_auc_mean | PASS | recomputed 0.8781 vs report 0.878 |
| diff_random_mat_nograph_estrogen-beta | PASS | recomputed -0.0148 vs -0.015 |
| wilcoxon_p_random_mat_nograph_estrogen-beta | PASS | recomputed p=0.0001 vs 0.0001 |
| npairs_random_mat_nograph_estrogen-beta | PASS | 25 pairs |
| random_mat_nograph_metstab-high_roc_auc_mean | PASS | recomputed 0.7775 vs report 0.778 |
| diff_random_mat_nograph_metstab-high | PASS | recomputed -0.0135 vs -0.013 |
| wilcoxon_p_random_mat_nograph_metstab-high | PASS | recomputed p=0.0451 vs 0.0451 |
| npairs_random_mat_nograph_metstab-high | PASS | 25 pairs |
| random_mat_nograph_metstab-low_roc_auc_mean | PASS | recomputed 0.7546 vs report 0.755 |
| diff_random_mat_nograph_metstab-low | PASS | recomputed -0.0101 vs -0.01 |
| wilcoxon_p_random_mat_nograph_metstab-low | PASS | recomputed p=0.0393 vs 0.0393 |
| npairs_random_mat_nograph_metstab-low | PASS | 25 pairs |
| random_rf_bbbp_roc_auc_mean | PASS | recomputed 0.9215 vs report 0.921 |
| diff_random_rf_bbbp | PASS | recomputed 0.0297 vs 0.03 |
| wilcoxon_p_random_rf_bbbp | PASS | recomputed p=0.0000 vs 0.0 |
| npairs_random_rf_bbbp | PASS | 25 pairs |
| random_rf_esol_rmse_mean | PASS | recomputed 0.5742 vs report 0.574 |
| diff_random_rf_esol | PASS | recomputed 0.1888 vs 0.189 |
| wilcoxon_p_random_rf_esol | PASS | recomputed p=0.0000 vs 0.0 |
| npairs_random_rf_esol | PASS | 25 pairs |
| random_rf_freesolv_rmse_mean | PASS | recomputed 0.5884 vs report 0.588 |
| diff_random_rf_freesolv | PASS | recomputed 0.2198 vs 0.22 |
| wilcoxon_p_random_rf_freesolv | PASS | recomputed p=0.0000 vs 0.0 |
| npairs_random_rf_freesolv | PASS | 25 pairs |
| random_rf_estrogen-alpha_roc_auc_mean | PASS | recomputed 0.9681 vs report 0.968 |
| diff_random_rf_estrogen-alpha | PASS | recomputed 0.0150 vs 0.015 |
| wilcoxon_p_random_rf_estrogen-alpha | PASS | recomputed p=0.0000 vs 0.0 |
| npairs_random_rf_estrogen-alpha | PASS | 25 pairs |
| random_rf_estrogen-beta_roc_auc_mean | PASS | recomputed 0.9264 vs report 0.926 |
| diff_random_rf_estrogen-beta | PASS | recomputed 0.0335 vs 0.034 |
| wilcoxon_p_random_rf_estrogen-beta | PASS | recomputed p=0.0000 vs 0.0 |
| npairs_random_rf_estrogen-beta | PASS | 25 pairs |
| random_rf_metstab-high_roc_auc_mean | PASS | recomputed 0.9016 vs report 0.902 |
| diff_random_rf_metstab-high | PASS | recomputed 0.1106 vs 0.111 |
| wilcoxon_p_random_rf_metstab-high | PASS | recomputed p=0.0000 vs 0.0 |
| npairs_random_rf_metstab-high | PASS | 25 pairs |
| random_rf_metstab-low_roc_auc_mean | PASS | recomputed 0.8732 vs report 0.873 |
| diff_random_rf_metstab-low | PASS | recomputed 0.1085 vs 0.108 |
| wilcoxon_p_random_rf_metstab-low | PASS | recomputed p=0.0000 vs 0.0 |
| npairs_random_rf_metstab-low | PASS | 25 pairs |
| random_svm_bbbp_roc_auc_mean | PASS | recomputed 0.9211 vs report 0.921 |
| diff_random_svm_bbbp | PASS | recomputed 0.0294 vs 0.029 |
| wilcoxon_p_random_svm_bbbp | PASS | recomputed p=0.0001 vs 0.0001 |
| npairs_random_svm_bbbp | PASS | 25 pairs |
| random_svm_esol_rmse_mean | PASS | recomputed 0.4854 vs report 0.485 |
| diff_random_svm_esol | PASS | recomputed 0.1000 vs 0.1 |
| wilcoxon_p_random_svm_esol | PASS | recomputed p=0.0000 vs 0.0 |
| npairs_random_svm_esol | PASS | 25 pairs |
| random_svm_freesolv_rmse_mean | PASS | recomputed 0.4714 vs report 0.471 |
| diff_random_svm_freesolv | PASS | recomputed 0.1028 vs 0.103 |
| wilcoxon_p_random_svm_freesolv | PASS | recomputed p=0.0000 vs 0.0 |
| npairs_random_svm_freesolv | PASS | 25 pairs |
| random_svm_estrogen-alpha_roc_auc_mean | PASS | recomputed 0.9686 vs report 0.969 |
| diff_random_svm_estrogen-alpha | PASS | recomputed 0.0155 vs 0.016 |
| wilcoxon_p_random_svm_estrogen-alpha | PASS | recomputed p=0.0000 vs 0.0 |
| npairs_random_svm_estrogen-alpha | PASS | 25 pairs |
| random_svm_estrogen-beta_roc_auc_mean | PASS | recomputed 0.9292 vs report 0.929 |
| diff_random_svm_estrogen-beta | PASS | recomputed 0.0363 vs 0.036 |
| wilcoxon_p_random_svm_estrogen-beta | PASS | recomputed p=0.0000 vs 0.0 |
| npairs_random_svm_estrogen-beta | PASS | 25 pairs |
| random_svm_metstab-high_roc_auc_mean | PASS | recomputed 0.9038 vs report 0.904 |
| diff_random_svm_metstab-high | PASS | recomputed 0.1128 vs 0.113 |
| wilcoxon_p_random_svm_metstab-high | PASS | recomputed p=0.0000 vs 0.0 |
| npairs_random_svm_metstab-high | PASS | 25 pairs |
| random_svm_metstab-low_roc_auc_mean | PASS | recomputed 0.8744 vs report 0.874 |
| diff_random_svm_metstab-low | PASS | recomputed 0.1096 vs 0.11 |
| wilcoxon_p_random_svm_metstab-low | PASS | recomputed p=0.0000 vs 0.0 |
| npairs_random_svm_metstab-low | PASS | 25 pairs |
| abl_random_bbbp_graph_structure | PASS | recomputed 0.0092 vs 0.009 |
| abl_random_bbbp_3D_distances | PASS | recomputed 0.0051 vs 0.005 |
| abl_random_bbbp_self-attention | PASS | recomputed 0.0366 vs 0.037 |
| abl_random_esol_graph_structure | PASS | recomputed -0.0254 vs -0.025 |
| abl_random_esol_3D_distances | PASS | recomputed -0.0055 vs -0.006 |
| abl_random_esol_self-attention | PASS | recomputed -0.0675 vs -0.068 |
| abl_random_freesolv_graph_structure | PASS | recomputed -0.0074 vs -0.007 |
| abl_random_freesolv_3D_distances | PASS | recomputed -0.0113 vs -0.011 |
| abl_random_freesolv_self-attention | PASS | recomputed -0.0891 vs -0.089 |
| abl_random_estrogen-alpha_graph_structure | PASS | recomputed 0.0060 vs 0.006 |
| abl_random_estrogen-alpha_3D_distances | PASS | recomputed 0.0014 vs 0.001 |
| abl_random_estrogen-alpha_self-attention | PASS | recomputed 0.0280 vs 0.028 |
| abl_random_estrogen-beta_graph_structure | PASS | recomputed 0.0148 vs 0.015 |
| abl_random_estrogen-beta_3D_distances | PASS | recomputed 0.0046 vs 0.005 |
| abl_random_estrogen-beta_self-attention | PASS | recomputed 0.0406 vs 0.041 |
| abl_random_metstab-high_graph_structure | PASS | recomputed 0.0135 vs 0.013 |
| abl_random_metstab-high_3D_distances | PASS | recomputed 0.0022 vs 0.002 |
| abl_random_metstab-high_self-attention | PASS | recomputed 0.0575 vs 0.057 |
| abl_random_metstab-low_graph_structure | PASS | recomputed 0.0101 vs 0.01 |
| abl_random_metstab-low_3D_distances | PASS | recomputed -0.0067 vs -0.007 |
| abl_random_metstab-low_self-attention | PASS | recomputed 0.1201 vs 0.12 |
| scaffold_gcn_bbbp_roc_auc_mean | PASS | recomputed 0.8592 vs report 0.859 |
| diff_scaffold_gcn_bbbp | PASS | recomputed -0.0173 vs -0.017 |
| wilcoxon_p_scaffold_gcn_bbbp | PASS | recomputed p=0.0020 vs 0.002 |
| npairs_scaffold_gcn_bbbp | PASS | 25 pairs |
| scaffold_gcn_esol_rmse_mean | PASS | recomputed 0.5104 vs report 0.51 |
| diff_scaffold_gcn_esol | PASS | recomputed 0.0109 vs 0.011 |
| wilcoxon_p_scaffold_gcn_esol | PASS | recomputed p=0.1645 vs 0.1645 |
| npairs_scaffold_gcn_esol | PASS | 25 pairs |
| scaffold_gcn_freesolv_rmse_mean | PASS | recomputed 0.4262 vs report 0.426 |
| diff_scaffold_gcn_freesolv | PASS | recomputed -0.0053 vs -0.005 |
| wilcoxon_p_scaffold_gcn_freesolv | PASS | recomputed p=0.4908 vs 0.4908 |
| npairs_scaffold_gcn_freesolv | PASS | 25 pairs |
| scaffold_gcn_estrogen-alpha_roc_auc_mean | PASS | recomputed 0.9486 vs report 0.949 |
| diff_scaffold_gcn_estrogen-alpha | PASS | recomputed 0.0116 vs 0.012 |
| wilcoxon_p_scaffold_gcn_estrogen-alpha | PASS | recomputed p=0.0001 vs 0.0001 |
| npairs_scaffold_gcn_estrogen-alpha | PASS | 25 pairs |
| scaffold_gcn_estrogen-beta_roc_auc_mean | PASS | recomputed 0.8826 vs report 0.883 |
| diff_scaffold_gcn_estrogen-beta | PASS | recomputed 0.0305 vs 0.03 |
| wilcoxon_p_scaffold_gcn_estrogen-beta | PASS | recomputed p=0.0000 vs 0.0 |
| npairs_scaffold_gcn_estrogen-beta | PASS | 25 pairs |
| scaffold_gcn_metstab-high_roc_auc_mean | PASS | recomputed 0.7740 vs report 0.774 |
| diff_scaffold_gcn_metstab-high | PASS | recomputed 0.0225 vs 0.023 |
| wilcoxon_p_scaffold_gcn_metstab-high | PASS | recomputed p=0.0588 vs 0.0588 |
| npairs_scaffold_gcn_metstab-high | PASS | 25 pairs |
| scaffold_gcn_metstab-low_roc_auc_mean | PASS | recomputed 0.7839 vs report 0.784 |
| diff_scaffold_gcn_metstab-low | PASS | recomputed 0.0409 vs 0.041 |
| wilcoxon_p_scaffold_gcn_metstab-low | PASS | recomputed p=0.0006 vs 0.0006 |
| npairs_scaffold_gcn_metstab-low | PASS | 25 pairs |
| scaffold_hybrid_bbbp_roc_auc_mean | PASS | recomputed 0.8885 vs report 0.888 |
| diff_scaffold_hybrid_bbbp | PASS | recomputed 0.0120 vs 0.012 |
| wilcoxon_p_scaffold_hybrid_bbbp | PASS | recomputed p=0.0551 vs 0.0551 |
| npairs_scaffold_hybrid_bbbp | PASS | 25 pairs |
| scaffold_hybrid_esol_rmse_mean | PASS | recomputed 0.5896 vs report 0.59 |
| diff_scaffold_hybrid_esol | PASS | recomputed 0.0902 vs 0.09 |
| wilcoxon_p_scaffold_hybrid_esol | PASS | recomputed p=0.0001 vs 0.0001 |
| npairs_scaffold_hybrid_esol | PASS | 25 pairs |
| scaffold_hybrid_freesolv_rmse_mean | PASS | recomputed 0.4375 vs report 0.437 |
| diff_scaffold_hybrid_freesolv | PASS | recomputed 0.0060 vs 0.006 |
| wilcoxon_p_scaffold_hybrid_freesolv | PASS | recomputed p=0.7915 vs 0.7915 |
| npairs_scaffold_hybrid_freesolv | PASS | 25 pairs |
| scaffold_hybrid_estrogen-alpha_roc_auc_mean | PASS | recomputed 0.9478 vs report 0.948 |
| diff_scaffold_hybrid_estrogen-alpha | PASS | recomputed 0.0109 vs 0.011 |
| wilcoxon_p_scaffold_hybrid_estrogen-alpha | PASS | recomputed p=0.0007 vs 0.0007 |
| npairs_scaffold_hybrid_estrogen-alpha | PASS | 25 pairs |
| scaffold_hybrid_estrogen-beta_roc_auc_mean | PASS | recomputed 0.8887 vs report 0.889 |
| diff_scaffold_hybrid_estrogen-beta | PASS | recomputed 0.0365 vs 0.037 |
| wilcoxon_p_scaffold_hybrid_estrogen-beta | PASS | recomputed p=0.0000 vs 0.0 |
| npairs_scaffold_hybrid_estrogen-beta | PASS | 25 pairs |
| scaffold_hybrid_metstab-high_roc_auc_mean | PASS | recomputed 0.8509 vs report 0.851 |
| diff_scaffold_hybrid_metstab-high | PASS | recomputed 0.0994 vs 0.099 |
| wilcoxon_p_scaffold_hybrid_metstab-high | PASS | recomputed p=0.0000 vs 0.0 |
| npairs_scaffold_hybrid_metstab-high | PASS | 25 pairs |
| scaffold_hybrid_metstab-low_roc_auc_mean | PASS | recomputed 0.8053 vs report 0.805 |
| diff_scaffold_hybrid_metstab-low | PASS | recomputed 0.0623 vs 0.062 |
| wilcoxon_p_scaffold_hybrid_metstab-low | PASS | recomputed p=0.0000 vs 0.0 |
| npairs_scaffold_hybrid_metstab-low | PASS | 25 pairs |
| scaffold_mat_bbbp_roc_auc_mean | PASS | recomputed 0.8765 vs report 0.876 |
| scaffold_mat_esol_rmse_mean | PASS | recomputed 0.4994 vs report 0.499 |
| scaffold_mat_freesolv_rmse_mean | PASS | recomputed 0.4315 vs report 0.431 |
| scaffold_mat_estrogen-alpha_roc_auc_mean | PASS | recomputed 0.9369 vs report 0.937 |
| scaffold_mat_estrogen-beta_roc_auc_mean | PASS | recomputed 0.8521 vs report 0.852 |
| scaffold_mat_metstab-high_roc_auc_mean | PASS | recomputed 0.7515 vs report 0.751 |
| scaffold_mat_metstab-low_roc_auc_mean | PASS | recomputed 0.7430 vs report 0.743 |
| scaffold_mat_noattention_bbbp_roc_auc_mean | PASS | recomputed 0.8504 vs report 0.85 |
| diff_scaffold_mat_noattention_bbbp | PASS | recomputed -0.0261 vs -0.026 |
| wilcoxon_p_scaffold_mat_noattention_bbbp | PASS | recomputed p=0.0000 vs 0.0 |
| npairs_scaffold_mat_noattention_bbbp | PASS | 25 pairs |
| scaffold_mat_noattention_esol_rmse_mean | PASS | recomputed 0.5103 vs report 0.51 |
| diff_scaffold_mat_noattention_esol | PASS | recomputed 0.0109 vs 0.011 |
| wilcoxon_p_scaffold_mat_noattention_esol | PASS | recomputed p=0.4908 vs 0.4908 |
| npairs_scaffold_mat_noattention_esol | PASS | 25 pairs |
| scaffold_mat_noattention_freesolv_rmse_mean | PASS | recomputed 0.4769 vs report 0.477 |
| diff_scaffold_mat_noattention_freesolv | PASS | recomputed 0.0454 vs 0.045 |
| wilcoxon_p_scaffold_mat_noattention_freesolv | PASS | recomputed p=0.1135 vs 0.1135 |
| npairs_scaffold_mat_noattention_freesolv | PASS | 25 pairs |
| scaffold_mat_noattention_estrogen-alpha_roc_auc_mean | PASS | recomputed 0.9120 vs report 0.912 |
| diff_scaffold_mat_noattention_estrogen-alpha | PASS | recomputed -0.0249 vs -0.025 |
| wilcoxon_p_scaffold_mat_noattention_estrogen-alpha | PASS | recomputed p=0.0000 vs 0.0 |
| npairs_scaffold_mat_noattention_estrogen-alpha | PASS | 25 pairs |
| scaffold_mat_noattention_estrogen-beta_roc_auc_mean | PASS | recomputed 0.8333 vs report 0.833 |
| diff_scaffold_mat_noattention_estrogen-beta | PASS | recomputed -0.0188 vs -0.019 |
| wilcoxon_p_scaffold_mat_noattention_estrogen-beta | PASS | recomputed p=0.0096 vs 0.0096 |
| npairs_scaffold_mat_noattention_estrogen-beta | PASS | 25 pairs |
| scaffold_mat_noattention_metstab-high_roc_auc_mean | PASS | recomputed 0.6964 vs report 0.696 |
| diff_scaffold_mat_noattention_metstab-high | PASS | recomputed -0.0551 vs -0.055 |
| wilcoxon_p_scaffold_mat_noattention_metstab-high | PASS | recomputed p=0.0008 vs 0.0008 |
| npairs_scaffold_mat_noattention_metstab-high | PASS | 25 pairs |
| scaffold_mat_noattention_metstab-low_roc_auc_mean | PASS | recomputed 0.6720 vs report 0.672 |
| diff_scaffold_mat_noattention_metstab-low | PASS | recomputed -0.0710 vs -0.071 |
| wilcoxon_p_scaffold_mat_noattention_metstab-low | PASS | recomputed p=0.0000 vs 0.0 |
| npairs_scaffold_mat_noattention_metstab-low | PASS | 25 pairs |
| scaffold_mat_nodistance_bbbp_roc_auc_mean | PASS | recomputed 0.8733 vs report 0.873 |
| diff_scaffold_mat_nodistance_bbbp | PASS | recomputed -0.0032 vs -0.003 |
| wilcoxon_p_scaffold_mat_nodistance_bbbp | PASS | recomputed p=0.7310 vs 0.731 |
| npairs_scaffold_mat_nodistance_bbbp | PASS | 25 pairs |
| scaffold_mat_nodistance_esol_rmse_mean | PASS | recomputed 0.4932 vs report 0.493 |
| diff_scaffold_mat_nodistance_esol | PASS | recomputed -0.0062 vs -0.006 |
| wilcoxon_p_scaffold_mat_nodistance_esol | PASS | recomputed p=0.9158 vs 0.9158 |
| npairs_scaffold_mat_nodistance_esol | PASS | 25 pairs |
| scaffold_mat_nodistance_freesolv_rmse_mean | PASS | recomputed 0.4241 vs report 0.424 |
| diff_scaffold_mat_nodistance_freesolv | PASS | recomputed -0.0074 vs -0.007 |
| wilcoxon_p_scaffold_mat_nodistance_freesolv | PASS | recomputed p=0.2872 vs 0.2872 |
| npairs_scaffold_mat_nodistance_freesolv | PASS | 25 pairs |
| scaffold_mat_nodistance_estrogen-alpha_roc_auc_mean | PASS | recomputed 0.9322 vs report 0.932 |
| diff_scaffold_mat_nodistance_estrogen-alpha | PASS | recomputed -0.0047 vs -0.005 |
| wilcoxon_p_scaffold_mat_nodistance_estrogen-alpha | PASS | recomputed p=0.0755 vs 0.0755 |
| npairs_scaffold_mat_nodistance_estrogen-alpha | PASS | 25 pairs |
| scaffold_mat_nodistance_estrogen-beta_roc_auc_mean | PASS | recomputed 0.8583 vs report 0.858 |
| diff_scaffold_mat_nodistance_estrogen-beta | PASS | recomputed 0.0062 vs 0.006 |
| wilcoxon_p_scaffold_mat_nodistance_estrogen-beta | PASS | recomputed p=0.3123 vs 0.3123 |
| npairs_scaffold_mat_nodistance_estrogen-beta | PASS | 25 pairs |
| scaffold_mat_nodistance_metstab-high_roc_auc_mean | PASS | recomputed 0.7517 vs report 0.752 |
| diff_scaffold_mat_nodistance_metstab-high | PASS | recomputed 0.0002 vs 0.0 |
| wilcoxon_p_scaffold_mat_nodistance_metstab-high | PASS | recomputed p=0.8119 vs 0.8119 |
| npairs_scaffold_mat_nodistance_metstab-high | PASS | 25 pairs |
| scaffold_mat_nodistance_metstab-low_roc_auc_mean | PASS | recomputed 0.7451 vs report 0.745 |
| diff_scaffold_mat_nodistance_metstab-low | PASS | recomputed 0.0021 vs 0.002 |
| wilcoxon_p_scaffold_mat_nodistance_metstab-low | PASS | recomputed p=0.6150 vs 0.615 |
| npairs_scaffold_mat_nodistance_metstab-low | PASS | 25 pairs |
| scaffold_mat_nograph_bbbp_roc_auc_mean | PASS | recomputed 0.8746 vs report 0.875 |
| diff_scaffold_mat_nograph_bbbp | PASS | recomputed -0.0019 vs -0.002 |
| wilcoxon_p_scaffold_mat_nograph_bbbp | PASS | recomputed p=0.7510 vs 0.751 |
| npairs_scaffold_mat_nograph_bbbp | PASS | 25 pairs |
| scaffold_mat_nograph_esol_rmse_mean | PASS | recomputed 0.5242 vs report 0.524 |
| diff_scaffold_mat_nograph_esol | PASS | recomputed 0.0248 vs 0.025 |
| wilcoxon_p_scaffold_mat_nograph_esol | PASS | recomputed p=0.2635 vs 0.2635 |
| npairs_scaffold_mat_nograph_esol | PASS | 25 pairs |
| scaffold_mat_nograph_freesolv_rmse_mean | PASS | recomputed 0.4455 vs report 0.446 |
| diff_scaffold_mat_nograph_freesolv | PASS | recomputed 0.0140 vs 0.014 |
| wilcoxon_p_scaffold_mat_nograph_freesolv | PASS | recomputed p=0.0710 vs 0.071 |
| npairs_scaffold_mat_nograph_freesolv | PASS | 25 pairs |
| scaffold_mat_nograph_estrogen-alpha_roc_auc_mean | PASS | recomputed 0.9287 vs report 0.929 |
| diff_scaffold_mat_nograph_estrogen-alpha | PASS | recomputed -0.0082 vs -0.008 |
| wilcoxon_p_scaffold_mat_nograph_estrogen-alpha | PASS | recomputed p=0.0004 vs 0.0004 |
| npairs_scaffold_mat_nograph_estrogen-alpha | PASS | 25 pairs |
| scaffold_mat_nograph_estrogen-beta_roc_auc_mean | PASS | recomputed 0.8454 vs report 0.845 |
| diff_scaffold_mat_nograph_estrogen-beta | PASS | recomputed -0.0067 vs -0.007 |
| wilcoxon_p_scaffold_mat_nograph_estrogen-beta | PASS | recomputed p=0.2521 vs 0.2521 |
| npairs_scaffold_mat_nograph_estrogen-beta | PASS | 25 pairs |
| scaffold_mat_nograph_metstab-high_roc_auc_mean | PASS | recomputed 0.7356 vs report 0.736 |
| diff_scaffold_mat_nograph_metstab-high | PASS | recomputed -0.0159 vs -0.016 |
| wilcoxon_p_scaffold_mat_nograph_metstab-high | PASS | recomputed p=0.0160 vs 0.016 |
| npairs_scaffold_mat_nograph_metstab-high | PASS | 25 pairs |
| scaffold_mat_nograph_metstab-low_roc_auc_mean | PASS | recomputed 0.7260 vs report 0.726 |
| diff_scaffold_mat_nograph_metstab-low | PASS | recomputed -0.0170 vs -0.017 |
| wilcoxon_p_scaffold_mat_nograph_metstab-low | PASS | recomputed p=0.0483 vs 0.0483 |
| npairs_scaffold_mat_nograph_metstab-low | PASS | 25 pairs |
| scaffold_rf_bbbp_roc_auc_mean | PASS | recomputed 0.9017 vs report 0.902 |
| diff_scaffold_rf_bbbp | PASS | recomputed 0.0253 vs 0.025 |
| wilcoxon_p_scaffold_rf_bbbp | PASS | recomputed p=0.0000 vs 0.0 |
| npairs_scaffold_rf_bbbp | PASS | 25 pairs |
| scaffold_rf_esol_rmse_mean | PASS | recomputed 0.7726 vs report 0.773 |
| diff_scaffold_rf_esol | PASS | recomputed 0.2732 vs 0.273 |
| wilcoxon_p_scaffold_rf_esol | PASS | recomputed p=0.0000 vs 0.0 |
| npairs_scaffold_rf_esol | PASS | 25 pairs |
| scaffold_rf_freesolv_rmse_mean | PASS | recomputed 0.6479 vs report 0.648 |
| diff_scaffold_rf_freesolv | PASS | recomputed 0.2165 vs 0.216 |
| wilcoxon_p_scaffold_rf_freesolv | PASS | recomputed p=0.0000 vs 0.0 |
| npairs_scaffold_rf_freesolv | PASS | 25 pairs |
| scaffold_rf_estrogen-alpha_roc_auc_mean | PASS | recomputed 0.9524 vs report 0.952 |
| diff_scaffold_rf_estrogen-alpha | PASS | recomputed 0.0155 vs 0.016 |
| wilcoxon_p_scaffold_rf_estrogen-alpha | PASS | recomputed p=0.0000 vs 0.0 |
| npairs_scaffold_rf_estrogen-alpha | PASS | 25 pairs |
| scaffold_rf_estrogen-beta_roc_auc_mean | PASS | recomputed 0.8982 vs report 0.898 |
| diff_scaffold_rf_estrogen-beta | PASS | recomputed 0.0460 vs 0.046 |
| wilcoxon_p_scaffold_rf_estrogen-beta | PASS | recomputed p=0.0000 vs 0.0 |
| npairs_scaffold_rf_estrogen-beta | PASS | 25 pairs |
| scaffold_rf_metstab-high_roc_auc_mean | PASS | recomputed 0.8701 vs report 0.87 |
| diff_scaffold_rf_metstab-high | PASS | recomputed 0.1186 vs 0.119 |
| wilcoxon_p_scaffold_rf_metstab-high | PASS | recomputed p=0.0000 vs 0.0 |
| npairs_scaffold_rf_metstab-high | PASS | 25 pairs |
| scaffold_rf_metstab-low_roc_auc_mean | PASS | recomputed 0.8101 vs report 0.81 |
| diff_scaffold_rf_metstab-low | PASS | recomputed 0.0671 vs 0.067 |
| wilcoxon_p_scaffold_rf_metstab-low | PASS | recomputed p=0.0000 vs 0.0 |
| npairs_scaffold_rf_metstab-low | PASS | 25 pairs |
| scaffold_svm_bbbp_roc_auc_mean | PASS | recomputed 0.9041 vs report 0.904 |
| diff_scaffold_svm_bbbp | PASS | recomputed 0.0276 vs 0.028 |
| wilcoxon_p_scaffold_svm_bbbp | PASS | recomputed p=0.0000 vs 0.0 |
| npairs_scaffold_svm_bbbp | PASS | 25 pairs |
| scaffold_svm_esol_rmse_mean | PASS | recomputed 0.6314 vs report 0.631 |
| diff_scaffold_svm_esol | PASS | recomputed 0.1320 vs 0.132 |
| wilcoxon_p_scaffold_svm_esol | PASS | recomputed p=0.0000 vs 0.0 |
| npairs_scaffold_svm_esol | PASS | 25 pairs |
| scaffold_svm_freesolv_rmse_mean | PASS | recomputed 0.5767 vs report 0.577 |
| diff_scaffold_svm_freesolv | PASS | recomputed 0.1452 vs 0.145 |
| wilcoxon_p_scaffold_svm_freesolv | PASS | recomputed p=0.0000 vs 0.0 |
| npairs_scaffold_svm_freesolv | PASS | 25 pairs |
| scaffold_svm_estrogen-alpha_roc_auc_mean | PASS | recomputed 0.9541 vs report 0.954 |
| diff_scaffold_svm_estrogen-alpha | PASS | recomputed 0.0172 vs 0.017 |
| wilcoxon_p_scaffold_svm_estrogen-alpha | PASS | recomputed p=0.0000 vs 0.0 |
| npairs_scaffold_svm_estrogen-alpha | PASS | 25 pairs |
| scaffold_svm_estrogen-beta_roc_auc_mean | PASS | recomputed 0.8974 vs report 0.897 |
| diff_scaffold_svm_estrogen-beta | PASS | recomputed 0.0453 vs 0.045 |
| wilcoxon_p_scaffold_svm_estrogen-beta | PASS | recomputed p=0.0000 vs 0.0 |
| npairs_scaffold_svm_estrogen-beta | PASS | 25 pairs |
| scaffold_svm_metstab-high_roc_auc_mean | PASS | recomputed 0.8613 vs report 0.861 |
| diff_scaffold_svm_metstab-high | PASS | recomputed 0.1098 vs 0.11 |
| wilcoxon_p_scaffold_svm_metstab-high | PASS | recomputed p=0.0000 vs 0.0 |
| npairs_scaffold_svm_metstab-high | PASS | 25 pairs |
| scaffold_svm_metstab-low_roc_auc_mean | PASS | recomputed 0.8177 vs report 0.818 |
| diff_scaffold_svm_metstab-low | PASS | recomputed 0.0747 vs 0.075 |
| wilcoxon_p_scaffold_svm_metstab-low | PASS | recomputed p=0.0000 vs 0.0 |
| npairs_scaffold_svm_metstab-low | PASS | 25 pairs |
| abl_scaffold_bbbp_graph_structure | PASS | recomputed 0.0019 vs 0.002 |
| abl_scaffold_bbbp_3D_distances | PASS | recomputed 0.0032 vs 0.003 |
| abl_scaffold_bbbp_self-attention | PASS | recomputed 0.0261 vs 0.026 |
| abl_scaffold_esol_graph_structure | PASS | recomputed -0.0248 vs -0.025 |
| abl_scaffold_esol_3D_distances | PASS | recomputed 0.0062 vs 0.006 |
| abl_scaffold_esol_self-attention | PASS | recomputed -0.0109 vs -0.011 |
| abl_scaffold_freesolv_graph_structure | PASS | recomputed -0.0140 vs -0.014 |
| abl_scaffold_freesolv_3D_distances | PASS | recomputed 0.0074 vs 0.007 |
| abl_scaffold_freesolv_self-attention | PASS | recomputed -0.0454 vs -0.045 |
| abl_scaffold_estrogen-alpha_graph_structure | PASS | recomputed 0.0082 vs 0.008 |
| abl_scaffold_estrogen-alpha_3D_distances | PASS | recomputed 0.0047 vs 0.005 |
| abl_scaffold_estrogen-alpha_self-attention | PASS | recomputed 0.0249 vs 0.025 |
| abl_scaffold_estrogen-beta_graph_structure | PASS | recomputed 0.0067 vs 0.007 |
| abl_scaffold_estrogen-beta_3D_distances | PASS | recomputed -0.0062 vs -0.006 |
| abl_scaffold_estrogen-beta_self-attention | PASS | recomputed 0.0188 vs 0.019 |
| abl_scaffold_metstab-high_graph_structure | PASS | recomputed 0.0159 vs 0.016 |
| abl_scaffold_metstab-high_3D_distances | PASS | recomputed -0.0002 vs -0.0 |
| abl_scaffold_metstab-high_self-attention | PASS | recomputed 0.0551 vs 0.055 |
| abl_scaffold_metstab-low_graph_structure | PASS | recomputed 0.0170 vs 0.017 |
| abl_scaffold_metstab-low_3D_distances | PASS | recomputed -0.0021 vs -0.002 |
| abl_scaffold_metstab-low_self-attention | PASS | recomputed 0.0710 vs 0.071 |
| gap_gcn_bbbp | PASS | recomputed 0.0396 vs 0.04 |
| gap_gcn_esol | PASS | recomputed -0.0532 vs -0.053 |
| gap_gcn_freesolv | PASS | recomputed -0.0879 vs -0.088 |
| gap_gcn_estrogen-alpha | PASS | recomputed 0.0074 vs 0.007 |
| gap_gcn_estrogen-beta | PASS | recomputed 0.0299 vs 0.03 |
| gap_gcn_metstab-high | PASS | recomputed 0.0736 vs 0.074 |
| gap_gcn_metstab-low | PASS | recomputed 0.0632 vs 0.063 |
| gap_hybrid_bbbp | PASS | recomputed 0.0225 vs 0.023 |
| gap_hybrid_esol | PASS | recomputed -0.0983 vs -0.098 |
| gap_hybrid_freesolv | PASS | recomputed -0.0588 vs -0.059 |
| gap_hybrid_estrogen-alpha | PASS | recomputed 0.0209 vs 0.021 |
| gap_hybrid_estrogen-beta | PASS | recomputed 0.0333 vs 0.033 |
| gap_hybrid_metstab-high | PASS | recomputed 0.0474 vs 0.047 |
| gap_hybrid_metstab-low | PASS | recomputed 0.0582 vs 0.058 |
| gap_mat_bbbp | PASS | recomputed 0.0153 vs 0.015 |
| gap_mat_esol | PASS | recomputed -0.1140 vs -0.114 |
| gap_mat_freesolv | PASS | recomputed -0.0629 vs -0.063 |
| gap_mat_estrogen-alpha | PASS | recomputed 0.0162 vs 0.016 |
| gap_mat_estrogen-beta | PASS | recomputed 0.0408 vs 0.041 |
| gap_mat_metstab-high | PASS | recomputed 0.0395 vs 0.04 |
| gap_mat_metstab-low | PASS | recomputed 0.0217 vs 0.022 |
| gap_mat_noattention_bbbp | PASS | recomputed 0.0047 vs 0.005 |
| gap_mat_noattention_esol | PASS | recomputed -0.0574 vs -0.057 |
| gap_mat_noattention_freesolv | PASS | recomputed -0.0192 vs -0.019 |
| gap_mat_noattention_estrogen-alpha | PASS | recomputed 0.0131 vs 0.013 |
| gap_mat_noattention_estrogen-beta | PASS | recomputed 0.0190 vs 0.019 |
| gap_mat_noattention_metstab-high | PASS | recomputed 0.0372 vs 0.037 |
| gap_mat_noattention_metstab-low | PASS | recomputed -0.0274 vs -0.027 |
| gap_mat_nodistance_bbbp | PASS | recomputed 0.0134 vs 0.013 |
| gap_mat_nodistance_esol | PASS | recomputed -0.1022 vs -0.102 |
| gap_mat_nodistance_freesolv | PASS | recomputed -0.0442 vs -0.044 |
| gap_mat_nodistance_estrogen-alpha | PASS | recomputed 0.0194 vs 0.019 |
| gap_mat_nodistance_estrogen-beta | PASS | recomputed 0.0299 vs 0.03 |
| gap_mat_nodistance_metstab-high | PASS | recomputed 0.0371 vs 0.037 |
| gap_mat_nodistance_metstab-low | PASS | recomputed 0.0263 vs 0.026 |
| gap_mat_nograph_bbbp | PASS | recomputed 0.0080 vs 0.008 |
| gap_mat_nograph_esol | PASS | recomputed -0.1134 vs -0.113 |
| gap_mat_nograph_freesolv | PASS | recomputed -0.0696 vs -0.07 |
| gap_mat_nograph_estrogen-alpha | PASS | recomputed 0.0184 vs 0.018 |
| gap_mat_nograph_estrogen-beta | PASS | recomputed 0.0327 vs 0.033 |
| gap_mat_nograph_metstab-high | PASS | recomputed 0.0419 vs 0.042 |
| gap_mat_nograph_metstab-low | PASS | recomputed 0.0286 vs 0.029 |
| gap_rf_bbbp | PASS | recomputed 0.0197 vs 0.02 |
| gap_rf_esol | PASS | recomputed -0.1984 vs -0.198 |
| gap_rf_freesolv | PASS | recomputed -0.0595 vs -0.06 |
| gap_rf_estrogen-alpha | PASS | recomputed 0.0156 vs 0.016 |
| gap_rf_estrogen-beta | PASS | recomputed 0.0283 vs 0.028 |
| gap_rf_metstab-high | PASS | recomputed 0.0315 vs 0.031 |
| gap_rf_metstab-low | PASS | recomputed 0.0631 vs 0.063 |
| gap_svm_bbbp | PASS | recomputed 0.0171 vs 0.017 |
| gap_svm_esol | PASS | recomputed -0.1460 vs -0.146 |
| gap_svm_freesolv | PASS | recomputed -0.1054 vs -0.105 |
| gap_svm_estrogen-alpha | PASS | recomputed 0.0145 vs 0.015 |
| gap_svm_estrogen-beta | PASS | recomputed 0.0318 vs 0.032 |
| gap_svm_metstab-high | PASS | recomputed 0.0425 vs 0.043 |
| gap_svm_metstab-low | PASS | recomputed 0.0567 vs 0.057 |
| tracked_numbers_recomputed | PASS | 406 quantities recomputed |
| clf_metrics_reported | PASS |  |
| reg_metrics_reported | PASS |  |
| ci95_reported | PASS |  |
| paired_t_and_wilcoxon | PASS |  |
| pvalues_in_unit_interval | PASS |  |
| generalization_gap_reported | PASS |  |
| ablations_reported | PASS |  |
| hybrid_evaluated | PASS |  |
| gcn_rf_svm_evaluated | PASS |  |
| pretrained_vs_scratch_reported | PASS |  |
| attention_analysis_reported | PASS |  |
| figures_present | PASS | 18 figures |
| figure_linked_fig_ablations.png | PASS |  |
| figure_linked_fig_attention_classes_bbbp_random.png | PASS |  |
| figure_linked_fig_attention_classes_bbbp_scaffold.png | PASS |  |
| figure_linked_fig_attention_classes_esol_random.png | PASS |  |
| figure_linked_fig_attention_classes_esol_scaffold.png | PASS |  |
| figure_linked_fig_attention_examples_bbbp_random.png | PASS |  |
| figure_linked_fig_attention_examples_bbbp_scaffold.png | PASS |  |
| figure_linked_fig_attention_examples_esol_random.png | PASS |  |
| figure_linked_fig_attention_examples_esol_scaffold.png | PASS |  |
| figure_linked_fig_attention_heads_bbbp_random.png | PASS |  |
| figure_linked_fig_attention_heads_bbbp_scaffold.png | PASS |  |
| figure_linked_fig_attention_heads_esol_random.png | PASS |  |
| figure_linked_fig_attention_heads_esol_scaffold.png | PASS |  |
| figure_linked_fig_average_ranks.png | PASS |  |
| figure_linked_fig_generalization_gap.png | PASS |  |
| figure_linked_fig_performance_random.png | PASS |  |
| figure_linked_fig_performance_scaffold.png | PASS |  |
| figure_linked_fig_pretrained_vs_scratch.png | PASS |  |
| report_exists | PASS | 49588 chars |