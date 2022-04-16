import mmcv 
bop_ycb_idx2class={
        1: '002_master_chef_can', 
        2: '003_cracker_box',
        3: '004_sugar_box', 
        4: '005_tomato_soup_can',
        5: '006_mustard_bottle',
        6: '007_tuna_fish_can',
        7: '008_pudding_box', 
        8: '009_gelatin_box',
        9: '010_potted_meat_can', 
        10: '011_banana', 
        11: '019_pitcher_base', 
        12: '021_bleach_cleanser', 
        13: '024_bowl', 
        14: '025_mug', 
        15: '035_power_drill',
        16: '036_wood_block',
        17: '037_scissors', 
        18: '040_large_marker',
        19: '051_large_clamp',
        20: '052_extra_large_clamp',
        21: '061_foam_brick', 
    }
bop_ycb_class2idx = dict([[bop_ycb_idx2class[k],k ] for k in bop_ycb_idx2class.keys() ])


