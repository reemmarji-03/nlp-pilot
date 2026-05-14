from tabs.english.prediction_tab import cluster_labels


def test_cluster_labels_preserves_duplicate_point_indices():
    labels = cluster_labels(
        points=[[1.0, 1.0], [1.0, 1.0], [9.0, 9.0]],
        centers=[[0.0, 0.0], [10.0, 10.0]],
    )

    assert labels == [0, 0, 1]
    assert len(labels) == 3
