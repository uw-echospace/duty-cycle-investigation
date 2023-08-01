def get_config():
    return {
        "dc_color_mappings": {
            '1800of1800' : 'cyan',
            '60of360' : 'red',
            '300of1800' : 'orange'
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