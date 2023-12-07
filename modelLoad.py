import os

import torch
import networkx as nx
from dash import dcc, html, Input, Output, Dash, callback_context, State
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go
from matplotlib import pyplot as plt
import dash_bootstrap_components as dbc
from capsule import CapsuleModel, getMNIST
from threading import Thread


# TODO: Visualizing parametrs

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
            weight = edge[2]['weight']
            edge_hover_text = f'Weight: {weight:.4f}'  # format to 4 decimal places

            edge_traces.append(go.Scatter(
                x=[x0, x1, None], y=[y0, y1, None],
                line=dict(width=edge[2]['width'], color=edge[2]['color']),
                mode='lines',
                hoverinfo='text',
                text=edge_hover_text,  # Add hover text
                hovertemplate='%{text}<extra></extra>'  # Remove the trace name on hover
            ))

        node_x, node_y, node_color, node_info = [], [], [], []
        for node in G.nodes(data=True):
            x, y = pos[node[0]]
            node_x.append(x)
            node_y.append(y)
            node_color.append(node[1]['color'])
            # You can add more node information here
            node_info_text = f'{node[0]}<br>' + \
                             f'Total weight: {sum(G[node[0]][n]["weight"] for n in G[node[0]])}'
            node_info.append(node_info_text)

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            marker=dict(size=10, color=node_color),
            text=node_info,  # Add node information
            hoverinfo='text',
            hovertemplate='%{text}<extra></extra>'
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
        self.loading_complete = False
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CapsuleModel(capdim=[(72, 4), (10, 20)], conv=[8, 16], in_channels=1).to(self.device)
        self.model.load_state_dict(torch.load('0.mo'))
        self.model.eval()
        self.preloaded_data = []
        self.preloaded_images = []
        self.current_loading_index = 0

    def preload_data(self, batch_size=10):
        print("Preloading data")
        data = getMNIST(batch_size=batch_size)
        for images, labels in data["train"]:
            with torch.no_grad():
                images, labels = images.to(self.device), labels.to(self.device)
                for i in range(images.shape[0]):
                    print(i)
                    c = self.model.getC(images[i].unsqueeze(0))
                    caps = self.model(images[i].unsqueeze(0))
                    preds = (caps ** 2).sum(dim=-1).argmax(-1).cpu()
                    original_image = images[i].to('cpu').numpy().squeeze()

                    # Add the data to the preloaded lists
                    print(c[2].to('cpu').numpy().squeeze(), preds.item())
                    print("--------")
                    self.preloaded_data.append((c[2].to('cpu').numpy().squeeze(), preds.item()))
                    self.preloaded_images.append(original_image)

                    # Save the image using Matplotlib
                    plt.figure()
                    plt.imshow(original_image, cmap='gray')
                    plt.axis('off')  # Hide the axis

                    # Ensure the 'assets' directory exists
                    assets_dir = 'assets'
                    if not os.path.exists(assets_dir):
                        os.makedirs(assets_dir)

                    image_path = os.path.join(assets_dir, f"{self.current_loading_index}.png")
                    plt.savefig(image_path)
                    plt.close()  # Close the figure to free memory

                    # Increment the index and check if we've loaded enough images
                    self.current_loading_index += 1
                    if self.current_loading_index >= 10:
                        self.loading_complete = True
                        print(self.loading_complete)
                        return


class GraphApp:
    def __init__(self):
        self.visualizer = GraphVisualizer()
        self.network_manager = CapsuleNetworkManager()
        self.app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        self.setup_layout()
        self.setup_callbacks()
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
            ])
        ], className='main-container')

    # Inside the GraphApp class
    # Inside the GraphApp class
    def setup_callbacks(self):
        @self.app.callback(
            [
                Output('network-graph', 'figure'),
                Output('display-image', 'src')
            ],
            [
                Input('update-button', 'n_clicks'),
                Input('left-button', 'n_clicks'),
                Input('right-button', 'n_clicks'),
                Input('network-graph', 'clickData')
            ],
            [State('network-graph', 'figure')]  # Passing the current figure as a state
        )
        def update_graph(update_btn_n, left_btn_n, right_btn_n, click_data, current_figure):
            ctx = callback_context
            if not ctx.triggered:
                raise PreventUpdate

            # Determine which input was triggered
            trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

            if 'left-button' in trigger_id and self.current_image > 0:
                self.current_image -= 1
            elif 'right-button' in trigger_id and self.current_image < len(self.network_manager.preloaded_data) - 1:
                self.current_image += 1

            # Now fetch the new data for the current_image
            c, preds = self.network_manager.preloaded_data[self.current_image]
            print(self.current_image)
            # print(c, preds)
            # print("-------")
            # Visualize the graph with the new data
            self.visualizer.visualize_coupling_coefficients(c, preds)
            fig = self.visualizer.generate_plotly_graph(self.visualizer.G)

            # If there was a click on the graph, update the graph highlight
            if 'network-graph' in trigger_id and click_data:
                clicked_node = click_data['points'][0]['text']
                self.visualizer.update_graph_highlight(clicked_node)
                fig = self.visualizer.generate_plotly_graph(
                    self.visualizer.G)  # Re-generate the figure to apply highlight

            # Update the image displayed
            image_path = f"assets/{self.current_image}.png" if self.network_manager.loading_complete else None

            # Return the updated figure and image path
            return fig, image_path

    def run(self):
        preload_thread = Thread(target=self.network_manager.preload_data, args=(10,))
        preload_thread.start()

        # Start the Dash server
        self.app.run_server(debug=True)


if __name__ == "__main__":
    graph_app = GraphApp()
    graph_app.run()
