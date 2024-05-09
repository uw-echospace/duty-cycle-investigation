# For consistence in data visualization, duty-cycles will be mapped to specific colors.
DC_COLOR_MAPPINGS = {
                    '30of30' : 'cyan',
                    '1of6' : 'red',
                    '5of30' : 'orange',
                    '30of180' : 'purple',
                    '10of60' : 'pink'
                        }

# These are the 6 locations studied using Audiomoths in 2022
SITE_NAMES = {
            'Central' : "Central Pond",
            'Foliage' : "Foliage",
            'Carp' : "Carp Pond",
            'Telephone' : "Telephone Field"
                }


FREQUENCY_COLOR_MAPPINGS = {
                    'LF' : 'cyan',
                    'LF1' : 'cyan',
                    'HF1' : 'red',
                    'HF2' : 'yellow',
                    'HF' : 'orange',
                        }


FREQ_GROUPS = {
                'Carp' : {'': [0, 96000],
                          'LF1': [13000, 50000],
                          'HF1': [34000, 74000],
                          'HF2': [42000, 96000]},

                'Foliage' : {'': [0, 96000],
                          'LF1': [13000, 50000],
                          'HF1': [34000, 74000],
                          'HF2': [42000, 96000]},

                'Telephone' : {'': [0, 96000],
                          'LF1': [13000, 50000],
                          'HF1': [30000, 78000],
                          'HF2': [41000, 102000]},

                'Central' : {'': [0, 96000],
                          'LF1': [13000, 50000],
                          'HF1': [34000, 74000],
                          'HF2': [42000, 96000]}
                }

SEATTLE_LATITUDE = 47.655181
SEATTLE_LONGITUDE = -122.293123