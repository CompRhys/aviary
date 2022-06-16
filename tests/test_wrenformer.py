from examples.wrenformer import run_wrenformer


def test_wrenformer_regression(df_matbench_jdft2d):

    train_df = df_matbench_jdft2d.sample(frac=0.8, random_state=0)
    test_df = df_matbench_jdft2d.drop(train_df.index)

    test_metrics, *_ = run_wrenformer(
        run_name="wrenformer",
        train_df=train_df,
        test_df=test_df,
        target_col="exfoliation_en",
        task_type="regression",
        n_attn_layers=2,
        timestamp="",
        epochs=25,
    )

    assert test_metrics["mae"] < 45
    assert test_metrics["rmse"] < 100
    assert test_metrics["r2"] > -0.1


def test_wrenformer_classification(df_matbench_phonons_wyckoff):

    train_df = df_matbench_phonons_wyckoff.sample(frac=0.8, random_state=0)
    test_df = df_matbench_phonons_wyckoff.drop(train_df.index)

    test_metrics, _, _ = run_wrenformer(
        run_name="wrenformer-robust",
        train_df=train_df,
        test_df=test_df,
        target_col="phdos_clf",
        task_type="classification",
        timestamp="",
        epochs=10,
    )

    assert test_metrics["accuracy"] > 0.85
    assert test_metrics["rocauc"] > 0.9
