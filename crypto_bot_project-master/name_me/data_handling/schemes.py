###############################################################################
############################### IMPORTS #######################################
###############################################################################

#external imports

###############################################################################

# internal imports
from ..data_handling import transformers as tr

###############################################################################

# typing (external | internal)
from typing import List

from ..data_handling.transformers import DataTransformer

###############################################################################
###############################################################################
###############################################################################

def get_stages(uuid: str) -> List[List[DataTransformer]]:
    if uuid == '614e94a2-3a35-4341-ba25-ad2b4a8709e8':
        return [
            [
                tr.HeikinAshiTransformer(['high','low','open','close']),
                tr.MovingMean( 4),
                tr.MovingMean(10),
                tr.RSI( 7),
                tr.RSI(14),
                tr.LambdaTransformer(lambda col : col, name='copy')
            ],
            
            [
                tr.FilterKeysTransformer(
                    lambda col : 'volume' not in col and 'turnover' not in col and 'rsi' not in col,
                    tr.LambdaTransformer(lambda col : col.div(col.shift(1) + 2**(-10)), name='quot')
                ),
                tr.FilterKeysTransformer(
                    lambda col : 'volume' in col or 'turnover' in col or 'rsi' in col,
                    tr.LambdaTransformer(lambda col : col, name='copy'),   
                )
            ],
        ]
    elif uuid == '07ff9122-4a78-4a77-ae5d-8f7e3a183bcf':
        return [
            [
                tr.MovingMean( 4),
                tr.MovingMean(10),
                tr.RSI( 7),
                tr.RSI(14),
                tr.LambdaTransformer(lambda col : col, name='copy')
            ],
            
            [
                tr.FilterKeysTransformer(
                    lambda col : 'volume' not in col and 'turnover' not in col and 'rsi' not in col,
                    tr.LambdaTransformer(lambda col : col.div(col.shift(1) + 2**(-10)), name='quot')
                ),
                tr.FilterKeysTransformer(
                    lambda col : 'volume' in col or 'turnover' in col or 'rsi' in col,
                    tr.LambdaTransformer(lambda col : col, name='copy'),   
                )
            ],
            
            [
                tr.MovingMin( 4),
                tr.MovingMin(10),
                tr.MovingMax( 4),
                tr.MovingMax(10),    
                tr.LambdaTransformer(lambda col : col, name='copy')
            ],
            
            # [
            #     tr.AddShiftsTransformer([2,3,4,10])
            # ]
            [
                tr.Normalizer()    
            ]
        ]
    else:
        raise RuntimeError(f"Transformer with uuid '{uuid}' doesn't exist!")


def get_transformer_from_stages(stages: List[List[DataTransformer]]) -> DataTransformer:
        
    MAIN = tr.SequentialTransformer([
        tr.ParallelTransformer(stage) for stage in stages
    ])

    return tr.SequentialTransformer([
        tr.ParallelTransformer([
            # apply main transformer
            MAIN,
            # get private columns
            tr.SequentialTransformer([
                # get private columns without quots
                tr.ParallelTransformer([
                    tr.CopyTransformer(lambda col : True),
                    tr.HeikinAshiTransformer(
                        ['high', 'low', 'open', 'close']
                    ),
                    tr.FilterKeysTransformer(
                        lambda col : col in ['open', 'close'],
                        tr.ParallelTransformer([
                            tr.MovingMean( 2),
                            tr.MovingMean( 3),
                            tr.MovingMean( 4),
                            tr.MovingMean(10),
                            tr.MovingMean(30)
                        ])
                    )
                ]),
                # keep columns and add quots
                tr.ParallelTransformer([
                    # keep and rename
                    tr.SequentialTransformer([
                        tr.CopyTransformer(lambda col : True),
                        tr.RenameColumnsTransformer(
                            lambda col : f'__{col.upper()}'
                        )
                    ]),
                    # add quotients and rename
                    tr.SequentialTransformer([
                        tr.LambdaTransformer(
                            lambda col : col.div(col.shift(1)), name="quot"
                        ),
                        tr.RenameColumnsTransformer(
                            lambda col : f'__{col.upper()}'    
                        )
                    ])
                ])
            ])
        ]),
        # simplify names
        tr.RenameColumnsTransformer(
            lambda c : c.replace("_copy", "")    
        )
    ])

def get_transformer(uuid: str) -> DataTransformer():
    return get_transformer_from_stages(
        get_stages(uuid)
    )


if __name__ == "__main__":
    trans = get_transformer(
        '614e94a2-3a35-4341-ba25-ad2b4a8709e8'
    )