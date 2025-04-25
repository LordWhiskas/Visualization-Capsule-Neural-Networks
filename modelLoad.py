import base64
import os
import logging
import tempfile

import torch
import networkx as nx
import numpy as np
from dash import dcc, html, Input, Output, Dash, callback_context, State
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go
from matplotlib import pyplot as plt
import dash_bootstrap_components as dbc
from capsule import CapsuleModel, getMNIST
from threading import Thread

logging.basicConfig(level=logging.INFO)


class GraphVisualizer:
    def __init__(self):
        self.G = nx.Graph()

    def visualize_coupling_coefficients(self, c, capdim):
        """
        Создаёт граф с узлами и ребрами на основе коэффициентов связи (c) и конфигурации капсул (capdim).
        """
        self.G.clear()
        layers = []
        # Добавляем узлы по слоям
        for idx, (num_caps, cap_dim) in enumerate(capdim):
            layer_nodes = [f"L{idx}_N{i}" for i in range(num_caps)]
            layers.append(layer_nodes)
            node_color = 'blue' if idx < len(capdim) - 1 else 'red'
            self.G.add_nodes_from(layer_nodes, layer=f'{idx}', color=node_color)

        # Для каждого соединения между соседними слоями
        for idx, (layer, next_layer) in enumerate(zip(layers, layers[1:])):
            # Приводим матрицу коэффициентов к NumPy для векторизации
            coupling_matrix = np.array(c[idx])
            max_caps = max(len(layer), len(next_layer))
            dynamic_threshold = 1.0 / max_caps
            # Выбираем индексы, где вес больше порога
            indices = np.argwhere(coupling_matrix > dynamic_threshold)
            for i, j in indices:
                weight = coupling_matrix[i, j]
                self.G.add_edge(
                    layer[i],
                    next_layer[j],
                    weight=weight,
                    color='#333333',
                    width=5 * weight,
                    base_color='#333333',
                    base_width=5 * weight
                )
        # Расчет позиций для узлов (по слоям)
        for idx, layer in enumerate(layers):
            x_offset = -1 + 2 * idx / (len(layers) - 1)
            for i, node in enumerate(layer):
                y_position = i * (7.2 / len(layer))
                self.G.nodes[node]['pos'] = (x_offset, y_position)
        return self.G

    def update_graph_highlight(self, clicked_node):
        """
        Обновляет цвета и ширину ребер в графе для выделения узла, по которому кликнули.
        """
        clicked_node = clicked_node.split('<')[0]
        for u, v, data in self.G.edges(data=True):
            if clicked_node in (u, v):
                self.G[u][v]['color'] = '#00ff00'  # Яркий зеленый для выделения
                self.G[u][v]['width'] = data['base_width'] * 2
            else:
                self.G[u][v]['color'] = data['base_color']
                self.G[u][v]['width'] = data['base_width']

    @staticmethod
    def generate_plotly_graph(G):
        """
        Генерирует график Plotly на основе графа networkx.
        """
        pos = nx.get_node_attributes(G, 'pos')
        edge_traces = []
        for u, v, data in G.edges(data=True):
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            weight = data['weight']
            edge_hover_text = f'Weight to {v}: {weight:.4f}'
            edge_traces.append(go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                line=dict(width=data['width'], color=data['color']),
                mode='lines',
                hoverinfo='text',
                text=edge_hover_text,
                hovertemplate='%{text}<extra></extra>'
            ))
        node_x, node_y, node_color, node_info = [], [], [], []
        for node, attr in G.nodes(data=True):
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_color.append(attr['color'])
            total_weight = sum(weight for _, _, weight in G.edges(node, data='weight'))
            node_info_text = f'{node}<br>Total connected weight: {total_weight:.4f}'
            node_info.append(node_info_text)
        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers',
            marker=dict(size=10, color=node_color),
            text=node_info,
            hoverinfo='text',
            hovertemplate='%{text}<extra></extra>'
        )
        fig = go.Figure(
            data=edge_traces + [node_trace],
            layout=go.Layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                margin=dict(l=0, r=0, t=0, b=0)
            )
        )
        fig.update_layout(autosize=True)
        return fig


class CapsuleNetworkManager:
    def __init__(self):
        self.loading_complete = False
        self.preload_target = 10
        self.preloading_now = False
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.capdim = [(72, 4), (20, 15), (10, 20)]
        self.conv = [8, 16]  # <-- добавь это
        self.in_channels = 1  # <-- и это

        self.model = CapsuleModel(
            capdim=self.capdim,
            conv=self.conv,
            in_channels=self.in_channels
        ).to(self.device)

        self.model_loaded = False
        self.preloaded_data = []
        self.preloaded_images = []
        self.current_loading_index = 0
        self.activations = []


    def load_model_from_file(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file {path} not found.")
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()
        self.model_loaded = True
        logging.info(f"Model loaded from {path}")

    def get_primary_capsule_activation(self, capsule_idx):
        if self.activations is None or capsule_idx >= self.activations.shape[0]:
            return None
        return self.activations[capsule_idx].detach().cpu().numpy()

    def get_capsule_attention_map(self, capsule_idx):
        fmap = self.model.get_feature_map()  # shape: (B, C, H, W)
        shape = self.model.get_caps_spatial_shape()  # (H, W)
        if fmap is None or shape is None:
            return None
        fmap = fmap.squeeze(0)  # (C, H, W)
        H, W = shape  # Например, (2, 2)
        cap_dim = self.capdim[0][1]  # 4
        num_caps_per_spatial = fmap.shape[0] // cap_dim  # 72/4 = 18

        # Если capsule_idx приходит как номер от 0 до 71 (всего 72 капсулы),
        # определяем, к какому месту в пространственной сетке он относится:
        spatial_idx = capsule_idx // num_caps_per_spatial  # номер ячейки (от 0 до 3)
        capsule_within_cell = capsule_idx % num_caps_per_spatial  # индекс капсулы внутри ячейки

        # Переформатируем fmap: (C, H, W) -> (num_caps_per_spatial, cap_dim, H, W)
        fmap_reshaped = fmap.view(num_caps_per_spatial, cap_dim, H, W)  # (18, 4, 2, 2)
        # Выбираем конкретную капсулу:
        capsule_feature = fmap_reshaped[capsule_within_cell]  # (4, 2, 2)

        # Для визуализации вычислим норму по размерности капсулы:
        capsule_feature = capsule_feature.permute(1, 2, 0)  # (2, 2, 4)
        heat = torch.norm(capsule_feature, dim=-1)  # (2, 2)

        # Интерполяция до нужного размера (например, 28x28)
        heat = torch.nn.functional.interpolate(
            heat.unsqueeze(0).unsqueeze(0),
            size=(28, 28),
            mode='bilinear',
            align_corners=False
        )
        return heat.squeeze().numpy()

    def preload_data(self, batch_size=10):
        logging.info("Preloading data")
        self.preloading_now = True  # <<< ДОБАВИЛИ
        assets_dir = 'assets'
        if not os.path.exists(assets_dir):
            os.makedirs(assets_dir)
        data = getMNIST(batch_size=batch_size)
        for images, labels in data["train"]:
            with torch.no_grad():
                images, labels = images.to(self.device), labels.to(self.device)
                for i in range(images.shape[0]):
                    c = self.model.getC(images[i].unsqueeze(0))
                    caps = self.model(images[i].unsqueeze(0))
                    self.activations = self.model.get_activations().squeeze(0)
                    preds = (caps ** 2).sum(dim=-1).argmax(-1).cpu()
                    original_image = images[i].cpu().numpy().squeeze()
                    c_processed = [tensor.cpu().numpy().squeeze() for tensor in c]
                    self.preloaded_data.append((c_processed, preds.item()))
                    self.preloaded_images.append(original_image)

                    plt.figure()
                    plt.imshow(original_image, cmap='gray')
                    plt.axis('off')
                    image_path = os.path.join(assets_dir, f"{self.current_loading_index}.png")
                    plt.savefig(image_path)
                    plt.close()

                    self.current_loading_index += 1
                    if self.current_loading_index >= self.preload_target:
                        self.loading_complete = True
                        self.preloading_now = False  # <<< ДОБАВИЛИ
                        logging.info("Preloading complete")
                        return


class GraphApp:
    def __init__(self):
        self.visualizer = GraphVisualizer()
        self.network_manager = CapsuleNetworkManager()
        self.app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        self.current_image = 0
        self.setup_layout()
        self.setup_callbacks()

    def setup_layout(self):
        self.app.layout = html.Div([
            html.Div(id='left-side-elements', className='left-container', children=[
                html.Div([
                    html.Div(id='img-zoom-container', className='card img-zoom', children=[
                        html.H2("Image from dataset", id='text-image', className='card-title'),
                        html.Img(id='display-image', src='assets/0.png')
                    ]),
                    html.Div(id='heatmap-container', className='card img-zoom', children=[
                        html.H2("Capsule Activation", className='card-title'),
                        html.Img(id='capsule-heatmap', src='')
                    ]),
                    html.Div(className='update-button-container', children=[
                        html.Button('Update Graph', id='update-button', className='update-button-style'),
                    ]),
                    html.Div([
                        html.Button('←', id='left-button', className='arrow-button'),
                        html.Button('→', id='right-button', className='arrow-button')
                    ], className='button-container')
                ], className='image-and-button-container'),
            ]),

            html.Div(id='graph', children=[dcc.Graph(id='network-graph')], className='main-image'),

            html.Div(id='right-side-elements', className='right-container', children=[
                html.Div([
                    html.Div(id='upload-container', className='card img-zoom', children=[
                        html.H2("Load Model", className='card-title'),
                        dcc.Upload(
                            id='upload-model',
                            children=html.Div([
                                html.P("Drag and Drop or"),
                                html.A("Select Model File (.pt or .mo)")
                            ]),
                            className='upload-area',
                            multiple=False
                        )
                    ]),
                ], className='image-and-button-container'),
                html.Div(id='model-config-container', className='card img-zoom', children=[
                    html.H2("CapsNet Config", className='card-title'),

                    html.Div([
                        html.Label("Conv layers (comma-separated)", htmlFor="conv-input"),
                        dcc.Input(id="conv-input", type="text", value="8,16", debounce=True, style={'width': '100%'})
                    ], style={'marginBottom': '10px'}),

                    html.Div([
                        html.Label("CapDim [(caps, dim), ...]", htmlFor="capdim-input"),
                        dcc.Input(id="capdim-input", type="text", value="(72,4),(20,15),(10,20)", debounce=True,
                                  style={'width': '100%'})
                    ], style={'marginBottom': '10px'}),

                    html.Button('Create Model', id='create-model-button', className='update-button-style'),

                    html.Div(id='model-create-status', style={'marginTop': '10px', 'textAlign': 'center'})
                ]),
                html.Div(id='preload-progress-container', className='card img-zoom', children=[
                    html.H2("Data Preloading", className='card-title'),
                    dbc.Progress(id='preload-progress', value=1, striped=True, animated=True, style={'height': '30px'}),
                    html.Div(id='preload-status', style={'textAlign': 'center', 'marginTop': '10px'})
                ]),
                dcc.Interval(id='progress-interval', interval=500, n_intervals=0),
                html.Div(id='model-reset-container', className='card img-zoom', children=[
                    html.H2("Reset Model", className='card-title'),
                    html.Button("Reset Everything", id='reset-model-button', className='update-button-style'),
                    html.Div(id='reset-status', style={'marginTop': '10px', 'textAlign': 'center'})
                ])
            ])
        ], className='main-container')

    def setup_callbacks(self):
        @self.app.callback(
            [Output('network-graph', 'figure'),
             Output('display-image', 'src'),
             Output('capsule-heatmap', 'src')],
            [Input('update-button', 'n_clicks'),
             Input('left-button', 'n_clicks'),
             Input('right-button', 'n_clicks'),
             Input('network-graph', 'clickData')],
            [State('network-graph', 'figure')]
        )
        def update_graph(update_btn_n, left_btn_n, right_btn_n, click_data, current_figure):
            ctx = callback_context
            if not ctx.triggered:
                raise PreventUpdate

            trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
            heatmap_path = ""

            if 'left-button' in trigger_id and self.current_image > 0:
                self.current_image -= 1
            elif 'right-button' in trigger_id and self.current_image < len(self.network_manager.preloaded_data) - 1:
                self.current_image += 1

            c, preds = self.network_manager.preloaded_data[self.current_image]
            self.visualizer.visualize_coupling_coefficients(c, self.network_manager.capdim)

            if 'network-graph' in trigger_id and click_data:
                clicked_node = click_data['points'][0]['text']
                self.visualizer.update_graph_highlight(clicked_node)

                if clicked_node.startswith("L0_N"):
                    cap_id = clicked_node.split('<')[0]
                    cap_idx = int(cap_id.split("_N")[1])
                    attn_map = self.network_manager.get_capsule_attention_map(cap_idx)
                    original = self.network_manager.preloaded_images[self.current_image]

                    if attn_map is not None:
                        import matplotlib.pyplot as plt
                        import os
                        # Создаем фигуру и отображаем оригинальное изображение
                        plt.figure(figsize=(4, 4))
                        plt.imshow(original, cmap='gray')

                        # Отображаем хитмап с прозрачностью, чтобы было видно оригинальное изображение
                        heatmap_im = plt.imshow(attn_map, cmap='hot', alpha=0.6)

                        # Выключаем оси, чтобы они не мешали визуализации
                        plt.axis('off')

                        # Добавляем заголовок с информацией о капсуле
                        plt.title(f'Activation of Capsule {cap_idx}', fontsize=12)

                        # Добавляем цветовую шкалу и подписываем её
                        cbar = plt.colorbar(heatmap_im, fraction=0.046, pad=0.04)
                        cbar.set_label('Activation norm', fontsize=10)

                        # Сохраняем изображение
                        heatmap_path = f"assets/heatmap_{cap_idx}.png"
                        plt.savefig(heatmap_path, bbox_inches='tight', pad_inches=0)
                        plt.close()

            fig = self.visualizer.generate_plotly_graph(self.visualizer.G)
            image_path = f"assets/{self.current_image}.png" if self.network_manager.loading_complete else None
            return fig, image_path, heatmap_path

        @self.app.callback(
            Output('model-create-status', 'children'),
            Input('create-model-button', 'n_clicks'),
            State('conv-input', 'value'),
            State('capdim-input', 'value')
        )
        def create_model(n_clicks, conv_val, capdim_val):
            if not n_clicks:
                raise PreventUpdate

            try:
                # Парсим conv
                conv = [int(x.strip()) for x in conv_val.split(',') if x.strip().isdigit()]
                # Парсим capdim
                capdim = eval(f"[{capdim_val}]")  # Преобразуем текст в список кортежей

                self.network_manager.capdim = capdim
                self.network_manager.conv = conv
                self.network_manager.in_channels = 1

                self.network_manager.model = CapsuleModel(
                    capdim=self.network_manager.capdim,
                    conv=self.network_manager.conv,
                    in_channels=self.network_manager.in_channels
                ).to(self.network_manager.device)
                self.network_manager.model_loaded = False
                self.network_manager.preloaded_data.clear()
                self.network_manager.preloaded_images.clear()
                self.network_manager.current_loading_index = 0
                self.network_manager.loading_complete = False
                logging.info("Model created with user parameters.")
                return html.Div("Model created. Now upload weights or preload.")
            except Exception as e:
                logging.error(f"Error creating model: {str(e)}")
                return html.Div(f"Error: {str(e)}")

        @self.app.callback(
            Output('upload-model', 'children'),
            Input('upload-model', 'contents'),
            State('upload-model', 'filename')
        )
        def load_model(uploaded_content, filename):
            if uploaded_content is None:
                raise PreventUpdate

            content_type, content_string = uploaded_content.split(',')
            decoded = base64.b64decode(content_string)

            # Временно сохраняем загруженный файл
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[-1]) as tmp_file:
                tmp_file.write(decoded)
                tmp_model_path = tmp_file.name

            try:
                self.network_manager.load_model_from_file(tmp_model_path)
                # Можно сразу запустить предзагрузку данных после загрузки модели
                preload_thread = Thread(target=self.network_manager.preload_data, args=(10,))
                preload_thread.start()
                return html.Div(f'Model "{filename}" successfully loaded.')
            except Exception as e:
                logging.error(str(e))
                return html.Div(f'Failed to load model: {str(e)}')

        @self.app.callback(
            Output('reset-status', 'children'),
            Input('reset-model-button', 'n_clicks')
        )
        def reset_model(n_clicks):
            if not n_clicks:
                raise PreventUpdate

            # Пересоздаем модель по текущим настройкам
            self.network_manager.model = CapsuleModel(
                capdim=self.network_manager.capdim,
                conv=self.network_manager.conv,
                in_channels=self.network_manager.in_channels
            ).to(self.network_manager.device)

            self.network_manager.model_loaded = False
            self.network_manager.loading_complete = False
            self.network_manager.preloaded_data.clear()
            self.network_manager.preloaded_images.clear()
            self.network_manager.activations = []
            self.network_manager.current_loading_index = 0

            import glob
            for path in glob.glob("assets/*.png"):
                try:
                    os.remove(path)
                except Exception as e:
                    logging.warning(f"Couldn't remove file {path}: {e}")

            return "Model reset. Ready for new configuration."

        @self.app.callback(
            [Output('preload-progress', 'value'),
             Output('preload-progress', 'max'),
             Output('preload-status', 'children')],
            Input('progress-interval', 'n_intervals')
        )
        def update_progress(n):
            if not self.network_manager.model_loaded:
                return 0, 1, "Model not loaded"

            if self.network_manager.preloading_now:
                return (n * 10) % 100, 100, "Preloading data..."
            else:
                if self.network_manager.loading_complete:
                    return 100, 100, "Preloading complete ✅"
                else:
                    return 0, 100, "Waiting for preloading..."

    def run(self):
        self.app.run_server(debug=True, dev_tools_hot_reload=False)


if __name__ == "__main__":
    graph_app = GraphApp()
    graph_app.run()
