# For consistence in data visualization, duty-cycles will be mapped to specific colors.
DC_COLOR_MAPPINGS = {
                    '1800of1800' : 'cyan',
                    '60of360' : 'red',
                    '300of1800' : 'orange',
                        }

# These are the 6 locations studied using Audiomoths in 2022
SITE_NAMES = {
            'Central' : "Central Pond",
            'Foliage' : "Foliage",
            'Carp' : "Carp Pond",
            'Telephone' : "Telephone Field",
            'Opposite' : "Opposite Carp Pond",
            'Fallen' : "Fallen Tree"
                }

# These are the observed frequency ranges for the 2 major groups of bat calls we have observed
FREQ_GROUPS = {
                'lf_' : [13000, 43000],
                'hf_' : [33000, 96000],
                '' :[0, 125000]
                }