def get_config():
    return {
        "dc_color_mappings": {
            '1800e1800' : 'cyan',
            '60e360' : 'red',
            '300e1800' : 'orange'
            },

        "site_names": {
            'Central' : "Central Pond",
            'Foliage' : "Foliage",
            'Carp' : "Carp Pond",
            'Telephone' : "Telephone Field",
            'Opposite' : "Opposite Carp Pond",
            'Fallen' : "Fallen Tree"
            },

        "freq_groups": {
            'lf_' : [0, 46000],
            'hf_' : [35000, 125000],
            '' :[0, 125000]
            }
    } 