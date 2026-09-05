import os
import torch
import torchburn


def test_visualize_basic():
    model = torch.nn.Sequential(
        torch.nn.Linear(8, 16),
        torch.nn.ReLU(),
        torch.nn.Linear(16, 4),
    )
    x = torch.randn(2, 8)

    html_path = "test_graph_output.html"
    try:
        viz = torchburn.visualize(model, x, output_html=html_path, print_summary=False)

        assert viz.total_compute_nodes > 0
        assert viz.native_ratio >= 0.0
        assert len(viz.summary()) > 0
        assert "flowchart TD" in viz.to_mermaid()
        assert "<!DOCTYPE html>" in viz.to_html()
        assert os.path.exists(html_path)
    finally:
        if os.path.exists(html_path):
            os.remove(html_path)
