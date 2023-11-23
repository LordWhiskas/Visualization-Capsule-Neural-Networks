import torch
import time
import networkx as nx
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
from matplotlib import pyplot as plt

from capsule import CapsuleModel, getMNIST
from PIL import Image
import numpy as np


class GraphVisualizer:
    def __init__(self):
        self.G = nx.Graph()
        self.tmp = None

    def visualize_coupling_coefficients(self, c, preds):
        self.G.clear()
        first_layer_nodes = [f"F{i}" for i in range(72)]
        output_layer_nodes = [f"O{j}" for j in range(10)]
        self.G.add_nodes_from(first_layer_nodes, layer='first', color='blue')
        self.G.add_nodes_from(output_layer_nodes, layer='output', color='red')
        for i in range(72):
            for j in range(10):
                weight = c[i][j]
                if weight <= 0.0149:
                    continue
                # print(weight, i * 4 + j)
                self.G.add_edge(f"F{i}", f"O{j}", weight=weight, color='#70c4f9', width=5 * weight)

        pos = {f"F{i}": (-1, i * (7.2 / 72)) for i in range(72)}
        pos.update({f"O{j}": (1, j * (7.2 / 10)) for j in range(10)})

        for node, p in pos.items():
            self.G.nodes[node]['pos'] = p

        self.tmp = self.G.copy()

        return self.G

    def update_graph_highlight(self, clicked_node):
        for u, v in list(self.G.edges()):
            if clicked_node in (u, v):
                self.G[u][v]['color'] = 'green'
                self.G[u][v]['width'] *= 1.1
            else:
                self.G[u][v]['color'] = self.tmp[u][v]['color']
                self.G[u][v]['width'] = self.tmp[u][v]['width']

    @staticmethod
    def generate_plotly_graph(G):
        pos = nx.get_node_attributes(G, 'pos')
        edge_traces = []
        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_traces.append(go.Scatter(
                x=[x0, x1, None], y=[y0, y1, None],
                line=dict(width=edge[2]['width'], color=edge[2]['color']),
                mode='lines', hoverinfo='none'
            ))

        node_x, node_y, node_color = [], [], []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_color.append(G.nodes[node]['color'])

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            marker=dict(size=10, color=node_color),
            text=list(G.nodes()), hoverinfo='text'
        )

        fig = go.Figure(data=edge_traces + [node_trace], layout=go.Layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            margin=dict(l=0, r=0, t=0, b=0)
        ))
        fig.update_layout(autosize=True)
        return fig


class CapsuleNetworkManager:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CapsuleModel(capdim=[(72, 4), (10, 20)], conv=[8, 16], in_channels=1).to(self.device)
        self.model.load_state_dict(torch.load('0.mo'))
        self.model.eval()
        self.original_image = None
        self.image_name = None

    def get_data(self, batch_size=1):
        print("take new data")
        return getMNIST(batch_size=batch_size)

    def process_data(self, data, current_image):
        for images, labels in data["train"]:
            with torch.no_grad():
                images, labels = images.to(self.device), labels.to(self.device)
                single_image, single_label = images[current_image].unsqueeze(0), labels[current_image].unsqueeze(0)
                c = self.model.getC(single_image)
                caps = self.model(single_image)
                preds = (caps ** 2).sum(dim=-1).argmax(-1).cpu()
                self.original_image = single_image[0].to('cpu').numpy().squeeze()
                plt.figure()
                plt.imshow(self.original_image, cmap='gray')
                plt.title(f"MNIST Image {preds.item()}")
                plt.axis('off')
                self.image_name = f"assets/{preds.item()}.png"
                plt.savefig(self.image_name)
            break
        return c[2].to('cpu').numpy().squeeze(), preds.item()


class GraphApp:
    def __init__(self):
        self.visualizer = GraphVisualizer()
        self.network_manager = CapsuleNetworkManager()
        self.app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])
        self.setup_layout()
        self.setup_callbacks()
        self.data = self.network_manager.get_data(10)
        self.current_image = 0

    def setup_layout(self):

        self.app.layout = html.Div([
            html.Div(id='left-side-elements', className='left-container', children=[
                html.Div([
                    html.Div(id='img-zoom-container', className='card img-zoom', children=[
                        html.H2("Image from dataset", id='text-image', className='card-title'),
                        html.Img(id='display-image', src='assets/0.png')
                    ]),
                    html.Button('Update Graph', id='update-button', className='update-button-style'),
                    html.Div([
                        html.Button('←', id='left-button', className='arrow-button'),
                        html.Button('→', id='right-button', className='arrow-button')
                    ], className='button-container'
                    )
                ], className='image-and-button-container'),
            ]),
            html.Div(id='graph', children=[dcc.Graph(id='network-graph')], className='main-image'),
            html.Div(id='right-side-elements', className='right-container', children=[
                html.Div(id='performance_metric', className='card img-zoom', children=[
                    html.H2("Performance Metric", id='text-metric', className='card-title'),
                    html.Img(id='Metric-image', src='assets/metric.png')
                ])
            ]),
        ], className='main-container')

    def setup_callbacks(self):
        @self.app.callback(
            [
                Output('network-graph', 'figure'),  # Output for the graph
                Output('display-image', 'src')  # Output for the image
            ],
            [Input('update-button', 'n_clicks'),
             Input('network-graph', 'clickData'),
             Input('left-button', 'n_clicks'),
             Input('right-button', 'n_clicks')
             ]
        )
        def update_graph(update_button_n_clicks, network_clickData, left_button_clickData, right_button_clickData):
            if network_clickData:
                clicked_node = network_clickData['points'][0]['text']
                self.visualizer.update_graph_highlight(clicked_node)
            elif update_button_n_clicks:
                print("Updating graph")
                c, preds = self.network_manager.process_data(self.data, self.current_image)
                self.visualizer.visualize_coupling_coefficients(c, preds)
                print(self.current_image)
            elif left_button_clickData:
                if self.current_image > 0:
                    self.current_image -= 1
                    self.network_manager.process_data(self.data, self.current_image)
                    print(self.current_image)
            elif right_button_clickData:
                if self.current_image < 9:
                    self.current_image += 1
                    self.network_manager.process_data(self.data, self.current_image)
                    print(self.current_image)
            fig = self.visualizer.generate_plotly_graph(self.visualizer.G)
            src = f"{self.network_manager.image_name}"
            return fig, src

    def run(self):
        self.app.run_server(debug=True)


if __name__ == "__main__":
    graph_app = GraphApp()
    graph_app.run()
