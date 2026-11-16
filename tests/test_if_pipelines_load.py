import joblib

def test_pipeline_loading():

    pipelines = [
        "models/PIPELINE_PD_v014_LR_cal_none.pkl",
        "models/PIPELINE_PD_v014_LR_cal_shift-test.pkl",
        "models/PIPELINE_PD_v014_GB_cal_shift-test.pkl",
        "models/PIPELINE_LGD_v014_amount.pkl",
        "models/PIPELINE_LGD_v014_stochastic.pkl",
    ]

    for pipeline_path in pipelines:

        print('testing pipeline:', pipeline_path)

        pipeline = joblib.load(pipeline_path)

        assert pipeline is not None, f"Failed to load {pipeline_path}"