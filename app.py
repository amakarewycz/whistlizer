import base64
import os
import datetime

import numpy as np
from flask import Flask, send_from_directory
from dash import dcc, ALL
from dash import html

import dash_bootstrap_components as dbc
import dash_uploader as du

from dash_extensions.enrich import Dash, ServersideOutput, Output, Input, State
import plotly.figure_factory as ff

import io
import keras

from parsing import parseMinSec, stripComments
from util import smooth_confusion_matrix2, smooth_join2

MODEL = 'models/model_ffn_202272511446'

CUE_POINTS = 'cue-points'

global model
model = None


APP_ID = 'user_large_video'
LARGE_UPLOAD = f'{APP_ID}_large_upload'
LARGE_UPLOAD_FN_STORE = f'{APP_ID}_large_upload_fn_store'
GROUND_TRUTH = f'{APP_ID}_ground_truth'
PREDICTIONS = f'{APP_ID}_predictions'


import dash_player


positive_threshold_div = html.Div([
    dbc.Label("Positive Threshold", html_for="positive-threshold"),
    dcc.Input(id='positive-threshold', placeholder="Positive Threshold", type="number", value=0.6,
              max=1, min=0.001),
    dbc.FormText(
        "Value threshold .001 - 1 to compare vs. prediction probability score."
        "  If value is greater than the threshold, the prediction is Positive, otherwise Negative"
    ),
])

frame_threshold_div = html.Div([
    dbc.Label("Frame Threshold", html_for="frame-threshold"),
    dcc.Input(id='frame-threshold', placeholder="Frame Threshold", type="number", value=2, max=10,
              min=1),
    dbc.FormText(
        "# of Frames 1 -10 threshold when comparing predictions against ground truth to come up with outcomes"
    ),
])

cue_delay_div = html.Div([
    dbc.Label("Cue delay", html_for="relax-cue"),
    dcc.Input(id='relax-cue', placeholder="Cue delay", type="number", value=0, max=2,
              min=0),
    dbc.FormText(
        "Value from 0 - 2 seconds. Time subtracted from the predicted time when building the play list such"
        "that there is a delay before whistle.  Gives the listener a chance to get ready!"
    ),
    html.Div(id='number-out')
])


layout = html.Div([
    dcc.Store(id=LARGE_UPLOAD_FN_STORE),
    dcc.Store(id=GROUND_TRUTH),
    dcc.Store(id=PREDICTIONS),
    dcc.Store(id=CUE_POINTS),
    dcc.Store(id='last-clicks'),
    dcc.Store(id='last-values'),
    dcc.Store(id="labels"),
    dbc.Row([html.Img(id="whistle", src='assets/whistle.jpg', height=100), html.H1(children="Whistlizer")]),
    du.Upload(id=LARGE_UPLOAD, text="Drag and Drop Video to Upload", max_file_size=5120),
    html.Div(
        style={
            'width': '40%',
            'float': 'left',
            'margin': '0% 5% 1% 5%'
        },
        children=[
            dbc.Form([positive_threshold_div, frame_threshold_div, cue_delay_div]),
            dash_player.DashPlayer(
                id='video-player',
                # url='/static/5U5A9285.MOV',
                # url='/static/37d8abac-10b1-11ed-b4ed-6003089edf50/5U5A9285.MOV',
                # url='static/9c585012-10b3-11ed-9cf9-6003089edf50/5U5A9299.mov',
                # url='static/output.webm',
                # url='static/5U5A9285.mov',
                # url='static/IMAGE_2365.mkv',
                # url='static/5U5A9285.mkv',
                controls=True,
                width='512',
                seekTo=10
            ),
            html.Div(id='button-container', children=[]),
            html.Div(id='gt-upload', children=[
                # du.Upload(id=f'{APP_ID}_ground_truth_upload', max_file_size=5120),
                dcc.Upload(
                    id='upload-data',
                    children=html.Div([
                        'Drag and Drop or ',
                        html.A('Select Ground Truth File')
                    ]),
                    style={
                        'width': '100%',
                        'height': '60px',
                        'lineHeight': '60px',
                        'borderWidth': '1px',
                        'borderStyle': 'dashed',
                        'borderRadius': '5px',
                        'textAlign': 'center',
                        'margin': '10px'
                    },
                    # Allow multiple files to be uploaded
                    multiple=True
                ),
                html.Div(id='output-data-upload'),
                html.Button(id='calculate'),
                html.Div(id='confusion-matrix')
            ]),
        ])])

last_n_clicks = None
last_value = None


def load_model():
    global model
    model = keras.models.load_model(MODEL)


def add_dash(app):
    @app.callback(
        Output('video-player', 'seekTo'),
        Output('last-clicks', 'data'),
        Output('last-values', 'data'),
        Input({'type': 'cue-button', 'index': ALL}, 'n_clicks'),
        Input({'type': 'cue-button', 'index': ALL}, 'value'),
        State('last-clicks', 'data'),
        State('last-values', 'data'),
    )
    def update_time(n_clicks, value, last_n_clicks, last_value):
        """Compares the last clicks vs. current  to see which button was pressed, sets the seekTo based on that time
         These buttons are dynamic, and will adjust to predictions from the model as well as additions from ground truth"""
        if last_value is None:
            return 0, n_clicks, value

        count = 0
        for i in zip(last_n_clicks, n_clicks):
            if i[0] != i[1]:
                break
            count = count + 1

        return float(value[count]) if len(value) > count else 0, n_clicks, value

    @app.callback(
        Output('button-container', 'children'),
        Input('video-player', 'url'),
        State('button-container', 'children'),
        Input(CUE_POINTS, 'data'),
        Input('labels', 'data')
    )
    def display_cue_buttons(url, children, data, labels):
        p = data
        children.clear()

        count = 0
        for i in p:
            pretty_label = str(i) if len(labels) != len(data) else str(i) + " " + labels[count]
            button = html.Button(pretty_label,
                                 id={'type': 'cue-button',
                                             'index': pretty_label},
                                 value=str(i), n_clicks=0)
            count = count + 1
            v = dbc.Row(button)
            children.append(v)
        return children

    @du.callback(
        output=Output(LARGE_UPLOAD_FN_STORE, 'data'),
        id=LARGE_UPLOAD,
    )
    def get_a_list(filenames):
        return {i: filenames[i] for i in range(len(filenames))}


    @app.callback(
        [
            Output('confusion-matrix', 'children'),
            Output('labels', 'data'),
            Output(CUE_POINTS, 'data'),
            Output('number-out', 'children')
        ],
        [
            State(PREDICTIONS, 'data'),
            State(GROUND_TRUTH, 'data'),
            State(CUE_POINTS, 'data'),
            Input("calculate", "n_clicks"),
            Input("positive-threshold", "value"),
            Input("frame-threshold", "value"),
            Input("relax-cue", "value")
        ]
    )
    def set_ground_truth(predictions, ground_truth, cue_points, n_clicks, positive_threshold, frame_threshold, relax_cue ):
        no = f"positive-threshold: {positive_threshold}, frame-threshold:{frame_threshold}, relax-cue:{relax_cue}"

        if ground_truth is None or ground_truth == [] or predictions is None or predictions == []:
            print("Predictions or ground truth is null")
            return "", [], cue_points, no

        outcomes = np.array(smooth_join2(predictions, ground_truth, positive_threshold=positive_threshold, frame_threshold=frame_threshold))
        secs = [i - relax_cue for i in outcomes[:, 0].astype(float)]
        labels = [i[:2] for i in outcomes[:, 1]]

        z = smooth_confusion_matrix2(predictions, ground_truth, positive_threshold=positive_threshold, frame_threshold=frame_threshold)

        print(z)
        x = ['no whistle', 'whistle']
        y = ['no whistle', 'whistle']
        # change each element of z to type string for annotations
        z_text = [[str(y) for y in x] for x in z]
        # set up figure
        fig = ff.create_annotated_heatmap(z, x=x, y=y, annotation_text=z_text, colorscale='GnBu')
        # add title
        fig.update_layout(title_text='<i><b>Confusion matrix</b></i>',
                          #xaxis = dict(title='x'),
                          #yaxis = dict(title='x')
                          )

        # add custom xaxis title
        fig.add_annotation(dict(font=dict(color="black", size=14),
                                x=0.5,
                                y=-0.15,
                                showarrow=False,
                                text="Predicted value",
                                xref="paper",
                                yref="paper"))

        # add custom yaxis title
        fig.add_annotation(dict(font=dict(color="black", size=14),
                                x=-0.35,
                                y=0.5,
                                showarrow=False,
                                text="Real value",
                                textangle=-90,
                                xref="paper",
                                yref="paper"))

        # adjust margins to make room for yaxis title
        fig.update_layout(margin=dict(t=50, l=200))

        return dcc.Graph(id='confusion-matrix', figure=fig), labels, secs, no

    def parse_contents(contents, filename, date):
        try:
            content_type, content_string = contents.split(',')
            decoded = base64.b64decode(content_string)
            cc = io.StringIO(decoded.decode('utf-8'))

            lines = [parseMinSec(stripComments("#")(line.rstrip())) for line in cc]

        except Exception as e:
            print(e)
            return html.Div([
                'There was an error processing this file.'
            ]), []

        return html.Div([
            html.H5(filename),
            html.H6(datetime.datetime.fromtimestamp(date)),

            # For debugging, display the raw contents provided by the web browser
            # html.Div('Raw Content'),
            # html.Pre(contents[0:200] + '...', style={
            #     'whiteSpace': 'pre-wrap',
            #     'wordBreak': 'break-all'
            # }),
            # html.Pre(f"{lines}" + '...', style={
            #     'whiteSpace': 'pre-wrap',
            #     'wordBreak': 'break-all'
            # })
        ]), lines

    @app.callback(Output('output-data-upload', 'children'),
                  Output(GROUND_TRUTH, 'data'),
                  Input('upload-data', 'contents'),
                  State('upload-data', 'filename'),
                  State('upload-data', 'last_modified'))
    def update_output(list_of_contents, list_of_names, list_of_dates):
        try:
            if list_of_contents is not None:
                children_ground_truth = [
                    parse_contents(c, n, d) for c, n, d in
                    zip(list_of_contents, list_of_names, list_of_dates)]
                return [children_ground_truth[0][0]], children_ground_truth[0][1]
        except Exception as e:
            print(e)
            return "Error check logs", []
        return "", []

    @app.callback(
        [
            Output('video-player', 'url'),
            Output(CUE_POINTS, 'data'),
            Output(PREDICTIONS, 'data')
        ],
        [
            Input(LARGE_UPLOAD_FN_STORE, 'data'),
            Input("positive-threshold", "value"),
            Input("relax-cue", "value")
        ],
    )
    def set_video_on_player(dic_of_names, positive_threshold, relax_cue):
        if dic_of_names is None:
            return [None, [], []]

        global model
        if model is None:
            load_model()


        video = os.path.join("static", os.path.basename(dic_of_names['0']))

        # video = "static/output.webm"
        #
        from util import featurize, positive_frames_to_secs, smooth, sr, frame_size_seconds, hop_length, \
            hop_in_window_divisions, frame_length, frame_length_c
        import math
        import functools

        sr = 44100 / 2
        frame_size_seconds = 0.7
        frame_length = frame_size_seconds * sr
        hop_in_window_divisions = 2
        hop_length = frame_size_seconds / hop_in_window_divisions * sr
        frame_length_c = math.ceil(frame_length)
        frame_length = frame_length_c
        hop_length = math.floor(frame_length_c / hop_in_window_divisions)

        predictions = model.predict(featurize(video))

        p = [i - relax_cue for i in functools.reduce(smooth, positive_frames_to_secs(predictions, positive_threshold),
                                               [])]  # subtract relax_cue /  seconds from prediction so that the whistle can be clearly heard when clicking cue button

        import numpy as np
        predictions = np.array(predictions)
        predictions = predictions.flatten()

        return [os.path.join("/", "static", os.path.basename(dic_of_names['0'])) if dic_of_names is not None else None,
                p, predictions]

    return app


if __name__ == '__main__':

    external_stylesheets = [
        dbc.themes.BOOTSTRAP,
    ]

    server = Flask(__name__)
    app = Dash(__name__, server=server, external_stylesheets=external_stylesheets)
    app.config['suppress_callback_exceptions'] = True


    # @app.server.route('/downloads/<path:path>')
    # def serve_static(path):
    #     return send_from_directory(
    #         Path("downloads"), path, as_attachment=True
    #     )

    @server.route('/static/<path:path>')
    def serve_static(path):
        # this never stops at breakpoint, running in different process?
        # a couple of things, anything served from '/static/<path:path>' must be found in that directory, children will not be found ie. /static/uuid/some-file  but will be found in /static/some-file
        root_dir = os.getcwd()
        # root_dir = "/Users/alex/models/"
        print("HI")
        return send_from_directory(os.path.join(root_dir, 'static'), path)

    # du.configure_upload(app, Path.cwd() / Path("temp"))
    du.configure_upload(app, os.path.join(os.getcwd(), "static"), use_upload_id=False)

    app.layout = layout
    app = add_dash(app)
    app.run_server(debug=True)