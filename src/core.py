# For consistence in data visualization, duty-cycles will be mapped to specific colors.
DC_COLOR_MAPPINGS = {
                    '1800of1800' : 'cyan',
                    '1of6' : 'violet',
                    '5of30' : 'yellowgreen',
                    '10of60' : 'darkgreen',
                    '30of180' : 'purple',
                    '60of360' : 'red',
                    '100of600' : 'deeppink',
                    '120of720' : 'maroon',
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

EXAMPLE_FILES_to_FILEPATHS = {
                            'UBNA_010/20220826_043000' : f'../data/audiomoth_recordings/recover-20220828/UBNA_010/20220826_043000.WAV',
                            'UBNA_010/20220826_050000' : f'../data/audiomoth_recordings/recover-20220828/UBNA_010/20220826_050000.WAV',
                            'UBNA_010/20220826_053000' : f'../data/audiomoth_recordings/recover-20220828/UBNA_010/20220826_053000.WAV',
                            'UBNA_010/20220826_060000' : f'../data/audiomoth_recordings/recover-20220828/UBNA_010/20220826_060000.WAV',
                            'UBNA_010/20220826_063000' : f'../data/audiomoth_recordings/recover-20220828/UBNA_010/20220826_063000.WAV',
                            'UBNA_010/20220826_070000' : f'../data/audiomoth_recordings/recover-20220828/UBNA_010/20220826_070000.WAV',
                            'UBNA_010/20220826_073000' : f'../data/audiomoth_recordings/recover-20220828/UBNA_010/20220826_073000.WAV',
                            'UBNA_010/20220826_080000' : f'../data/audiomoth_recordings/recover-20220828/UBNA_010/20220826_080000.WAV',
                            'UBNA_010/20220826_083000' : f'../data/audiomoth_recordings/recover-20220828/UBNA_010/20220826_083000.WAV',
                            'UBNA_010/20220826_090000' : f'../data/audiomoth_recordings/recover-20220828/UBNA_010/20220826_090000.WAV',
                            'UBNA_010/20220826_093000' : f'../data/audiomoth_recordings/recover-20220828/UBNA_010/20220826_093000.WAV',
                            'UBNA_001/20220820_043000' : f'../data/audiomoth_recordings/recover-20220822/UBNA_001/20220820_043000.WAV',
                            'UBNA_001/20220820_050000' : f'../data/audiomoth_recordings/recover-20220822/UBNA_001/20220820_050000.WAV',
                            'UBNA_001/20220820_053000' : f'../data/audiomoth_recordings/recover-20220822/UBNA_001/20220820_053000.WAV',
                            'UBNA_001/20220820_060000' : f'../data/audiomoth_recordings/recover-20220822/UBNA_001/20220820_060000.WAV',
                            'UBNA_001/20220820_063000' : f'../data/audiomoth_recordings/recover-20220822/UBNA_001/20220820_063000.WAV',
                            'UBNA_001/20220820_070000' : f'../data/audiomoth_recordings/recover-20220822/UBNA_001/20220820_070000.WAV',
                            'UBNA_001/20220820_073000' : f'../data/audiomoth_recordings/recover-20220822/UBNA_001/20220820_073000.WAV',
                            'UBNA_001/20220820_080000' : f'../data/audiomoth_recordings/recover-20220822/UBNA_001/20220820_080000.WAV',
                            'UBNA_001/20220820_083000' : f'../data/audiomoth_recordings/recover-20220822/UBNA_001/20220820_083000.WAV',
                            'UBNA_001/20220820_090000' : f'../data/audiomoth_recordings/recover-20220822/UBNA_001/20220820_090000.WAV',
                            'UBNA_001/20220820_093000' : f'../data/audiomoth_recordings/recover-20220822/UBNA_001/20220820_093000.WAV',
                            'UBNA_002/20210910_030000': f'../data/audiomoth_recordings/recover-20210912/UBNA_002/20210910_030000.WAV'
                                }

EXAMPLE_FILES_to_DETECTIONS = {
                            'UBNA_010/20220826_043000' : f'../data/raw/Central/bd2__Central_20220826_043000.csv',
                            'UBNA_010/20220826_050000' : f'../data/raw/Central/bd2__Central_20220826_050000.csv',
                            'UBNA_010/20220826_053000' : f'../data/raw/Central/bd2__Central_20220826_053000.csv',
                            'UBNA_010/20220826_060000' : f'../data/raw/Central/bd2__Central_20220826_060000.csv',
                            'UBNA_010/20220826_063000' : f'../data/raw/Central/bd2__Central_20220826_063000.csv',
                            'UBNA_010/20220826_070000' : f'../data/raw/Central/bd2__Central_20220826_070000.csv',
                            'UBNA_010/20220826_073000' : f'../data/raw/Central/bd2__Central_20220826_073000.csv',
                            'UBNA_010/20220826_080000' : f'../data/raw/Central/bd2__Central_20220826_080000.csv',
                            'UBNA_010/20220826_083000' : f'../data/raw/Central/bd2__Central_20220826_083000.csv',
                            'UBNA_010/20220826_090000' : f'../data/raw/Central/bd2__Central_20220826_090000.csv',
                            'UBNA_010/20220826_093000' : f'../data/raw/Central/bd2__Central_20220826_093000.csv',
                            'UBNA_001/20220820_043000' : f'../data/raw/Telephone/bd2__Telephone_20220820_043000.csv',
                            'UBNA_001/20220820_050000' : f'../data/raw/Telephone/bd2__Telephone_20220820_050000.csv',
                            'UBNA_001/20220820_053000' : f'../data/raw/Telephone/bd2__Telephone_20220820_053000.csv',
                            'UBNA_001/20220820_060000' : f'../data/raw/Telephone/bd2__Telephone_20220820_060000.csv',
                            'UBNA_001/20220820_063000' : f'../data/raw/Telephone/bd2__Telephone_20220820_063000.csv',
                            'UBNA_001/20220820_070000' : f'../data/raw/Telephone/bd2__Telephone_20220820_070000.csv',
                            'UBNA_001/20220820_073000' : f'../data/raw/Telephone/bd2__Telephone_20220820_073000.csv',
                            'UBNA_001/20220820_080000' : f'../data/raw/Telephone/bd2__Telephone_20220820_080000.csv',
                            'UBNA_001/20220820_083000' : f'../data/raw/Telephone/bd2__Telephone_20220820_083000.csv',
                            'UBNA_001/20220820_090000' : f'../data/raw/Telephone/bd2__Telephone_20220820_090000.csv',
                            'UBNA_001/20220820_093000' : f'../data/raw/Telephone/bd2__Telephone_20220820_093000.csv',
                            'UBNA_002/20210910_030000': f'../batdetect2_outputs/recover-20210912/Foliage/bd2_20210910_030000.csv'
                                }

EXAMPLE_FILES_from_LOCATIONS = {
                                'Foliage' : 'UBNA_002/20210910_030000',
                                'Telephone' : 'UBNA_001/20220820_093000',
                                'Central' : 'UBNA_010/20220826_070000'
                                }