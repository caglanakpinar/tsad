import datetime
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import random
import datetime

from data_access import GetData
from utils import read_yaml, split_groups, get_results, date_part
from configs import conf


convert_date = lambda x:  datetime.datetime.strptime(str(x)[0:19], "%Y-%m-%d %H:%M:%S")
get_date = lambda date, date_col: str(date['points'][0]['customdata'])[0:10] if 'customdata' in date['points'][0].keys() else date['points'][0][date_col]


def adding_filter(filter_id, labels, size, is_multi_select, value):
    return html.Div([
        html.Div(filter_id, style={'width': '40%', 'float': 'left', 'display': 'inline-block'}),
        dcc.Dropdown(
            id=filter_id,
            options=[{'label': i, 'value': i} for i in labels],
            multi=True if is_multi_select else False,
            value=value
        )
    ],
        style={'width': str(size) + '%', 'display': 'inline-block'})


def adding_filter_to_pane(added_filters, f_style):
    return html.Div(added_filters, style=f_style)


def adding_plots_to_pane(plot_id, hover_data, size):
    return html.Div([
        dcc.Graph(
            id=plot_id,
            hoverData={'points': [hover_data]}
        )
    ], style={'width': str(size) + '%', 'display': 'inline-block', 'padding': '0 90'})


def get_descriptives(data, feature):
    output = []
    for metric in [(np.mean, 'mean'), (max, 'max'), (min, 'min'), (np.median, 'median'), (np.median, 'median')]:
        output.append({'value': metric[0](list(data[data[feature] == data[feature]][feature])), 'metric': metric[1]})
    return pd.DataFrame(output)


def data_source():
    jobs = read_yaml(conf('docs_main_path'), 'ml_execute.yaml')
    model_infos = jobs[list(jobs.keys())[0]]['execute'][0]['params']
    try:
        source = GetData(data_query_path="sample_data.csv",
                         data_source="csv",
                         time_indicator=model_infos['time_indicator'],
                         feature=model_infos['feature'], test=1000)
        source.query_data_source()
        source.convert_feature()
        data = source.data
    except Exception as e:
        data = pd.DataFrame()
        print("no data is available")
    return data


def check_for_time_part_groups_on_data(data, t_dimensions, time_indicator):
    if len(t_dimensions) != 0:
        for t_dimension in t_dimensions:
            data[t_dimension] = data[time_indicator].apply(lambda x: date_part(convert_date(x), t_dimension))
    return data


def get_filters(data):
    jobs = read_yaml(conf('docs_main_path'), 'ml_execute.yaml')
    #model_conf = read_yaml(conf('model_main_path'), 'model_configuration.yaml')
    model_infos = jobs[list(jobs.keys())[0]]['execute'][0]['params']
    groups, date_col, feature = split_groups(model_infos['groups']), model_infos['time_indicator'], model_infos['feature']
    #t_dimensions = model_conf['infos']['time_groups'].split("*")
    #data = check_for_time_part_groups_on_data(data, t_dimensions, date_col)
    #groups += t_dimensions
    if groups not in ['None', None, []]:
        if len(groups) > 3:
            groups = random.sample(groups, 3)
    else:
        groups = list(set(list(data.columns)) - set([date_col, feature, 'Unnamed: 0.1', 'Unnamed: 0']))
        if groups >= 4:
            groups = list(filter(lambda col: type(col) == str, groups))
            if len(groups) >= 4:
                groups = random.sample(groups, 3) if len(groups) > 3 else [date_col]
    print("filters :", groups)
    num_f_p = len(groups)
    filter_datas = []
    for g in groups:
        filter_datas.append(list(data[data[g] == data[g]][g].unique()) + ['ALL'])
    filter_ids = groups

    filter_sizes = [30] * num_f_p
    multiple_selection = [False] * num_f_p
    values = ['ALL'] * num_f_p
    filters = list(zip(filter_ids, filter_datas, filter_sizes, multiple_selection, values))
    hover_data = [{date_col: min(data[model_infos['time_indicator']])}] * 3
    return num_f_p, filters, hover_data, groups, filter_ids, date_col, feature, data


def get_updated_filters(filter, filter_index):
    opts = filter[4][filter_index]
    return [{'label': opt, 'value': opt} for opt in opts]


def create_dashboard(server):
    external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
    app = dash.Dash(__name__, server=server, external_stylesheets=external_stylesheets, routes_pathname_prefix='/dash/')
    try:
        data = data_source()
        num_f_p, filters, hover_datas, groups, filter_ids, date_col, feature, data = get_filters(data)
        app.layout = html.Div()
        if len(data) == 0:
            return app
    except Exception as e:
        print(e)
        app.layout = html.Div()
        return app

    filter_style = {
                    'borderBottom': 'thin lightgrey solid',
                    'backgroundColor': 'rgb(250, 250, 250)',
                    'padding': '10px 5px'
    }
    plot_ids = ['line_chart', 'anomaly', 'daiy_numbers']
    plot_sizes = [99, 45, 45]
    plot_dfs = []
    plots = list(zip(plot_ids, plot_sizes, hover_datas))
    # adding filters
    pane_count = int(len(filters) / num_f_p) if int(len(filters) / num_f_p) == len(filters) / num_f_p else int(
        len(filters) / num_f_p) + 1
    components = []
    for i in range(pane_count):
        _filters = filters[i * num_f_p:(i + 1) * num_f_p] if i != pane_count - 1 else filters[i * num_f_p:]
        _pane = adding_filter_to_pane([adding_filter(f[0], f[1], f[2], f[3], f[4]) for f in _filters], filter_style)
        components.append(_pane)
    # adding plots
    for p in plots:
        components.append(adding_plots_to_pane(p[0], p[2], p[1]))
    app.layout = html.Div(components)

    @app.callback(
        dash.dependencies.Output(plot_ids[0], 'figure'),
        [dash.dependencies.Input(f, 'value') for f in get_filters(data_source())[4]]
    )
    def update_graph(*args):
        data = data_source()
        filter_ids, date_col, feature = get_filters(data)[4:-1]
        type_converter = lambda filter: "'" if type(list(data[filter])[0]) == str else " "
        for f in range(len(filter_ids)):
            filter = filter_ids[f]
            conv = type_converter(filter)
            query_str = ""
            if args[f] != 'ALL':
                is_started = " " if query_str == "" else " and "
                query_str += is_started + " " + filter + " == " + conv + str(args[f]) + conv

        dff = data if query_str == "" else data.query(query_str)
        dff = dff.pivot_table(index=date_col, aggfunc={feature: 'mean'}).reset_index()
        trace = [go.Scatter(x=dff[date_col], y=dff[m[0]], mode='markers+lines',
                            customdata=dff[date_col], name=m[1]+m[0]) for m in [(feature, 'Actual ')]]
        #trace = [go.Scatter(x=dff[date_col], y=dff[m[0]], mode='markers+lines',
        #                    customdata=dff[date_col], name=m[1]+m[0]) for m in [('KisiSayimi', 'Actual ')]]
        return {"data": trace,  "layout": go.Layout(height=600, title="Time Line Of " + feature)}

    @app.callback(
        dash.dependencies.Output(plot_ids[1], 'figure'),
        [dash.dependencies.Input(f, 'value') for f in get_filters(data_source())[4]]
    )
    def update_graph(*args):
        data = data_source()
        filter_ids, date_col, feature = get_filters(data)[4:-1]
        type_converter = lambda filter: "'" if type(list(data[filter])[0]) == str else " "
        for f in range(len(filter_ids)):
            filter = filter_ids[f]
            conv = type_converter(filter)
            query_str = ""
            if args[f] != 'ALL':
                is_started = " " if query_str == "" else " and "
                query_str += is_started + " " + filter + " == " + conv + str(args[f]) + conv
        dff = data if query_str == "" else data.query(query_str)
        descriptives = get_descriptives(dff, feature)

        trace = [go.Bar(x=descriptives['metric'], y=descriptives['value'])]
        return {"data": trace, "layout": go.Layout(height=600, title=feature + " Descriptives")}

    @app.callback(
        dash.dependencies.Output(plot_ids[2], 'figure'),
        [dash.dependencies.Input(f, 'value') for f in get_filters(data_source())[4]]
    )
    def update_graph(*args):
        data = data_source()
        filter_ids, date_col, feature = get_filters(data)[4:-1]
        type_converter = lambda filter: "'" if type(list(data[filter])[0]) == str else " "
        for f in range(len(filter_ids)):
            filter = filter_ids[f]
            conv = type_converter(filter)
            query_str = ""
            if args[f] != 'ALL':
                is_started = " " if query_str == "" else " and "
                query_str += is_started + " " + filter + " == " + conv + str(args[f]) + conv

        result_data = get_results(date_col)
        dff = result_data if query_str == "" else result_data.query(query_str)
        if len(dff) == 0:
            return {"data": [], "layout": go.Layout(height=600, title="Time Line Of " + feature)}
        else:
            trace = [go.Scatter(x=dff[date_col],
                                y=dff['predict'],
                                mode='markers+lines',
                                customdata=dff[date_col],
                                name='prediction'),
                    go.Bar(x=dff[date_col], y=dff['predicted_label'])]
            return {"data": trace,
                    "layout": go.Layout(height=600, title="Anomaly Detection and Prediction " + feature)}


    #@app.callback(
    #    dash.dependencies.Output(plot_ids[2], 'figure'),
    #    [dash.dependencies.Input(f, 'value') for f in filter_ids] +
    #    [dash.dependencies.Input(plot_ids[0], "hoverData")]
    #)
    #def update_graph(franchise_id, city, isoweekday, date):
    #    query_str = ""
    #    # franchise_id
    #    if franchise_id != 'ALL':
    #        is_started = " " if query_str == "" else " and "
    #        query_str += is_started + " BayiKodu == '" + str(franchise_id) + "' "
    #    # isoweekday
    #    if isoweekday != 'ALL':
    #        isoweekday_str = str(isoweekday)
    #        is_started = " " if query_str == "" else " and "
    #        query_str += is_started + " isoweekday == " + isoweekday_str + " "
    #    # City
    #    if city != 'ALL':
    #        is_started = " " if query_str == "" else " and "
    #        query_str += is_started + " City == '" + city + "' "
    #    # date
    #    is_started = " " if query_str == "" else " and "
    #    query_str += is_started + " " + date_col + " == '" + get_date(date) + "'"
    #    dff = data if query_str == "" else data.query(query_str)
    #    aggfunc ={get_col('ad_label', l): 'mean' for l in range(1, 4)}
    #    dff = dff.pivot_table(index='BayiKodu', aggfunc=aggfunc).reset_index()
    #    trace = [go.Bar(x=dff['BayiKodu'], y=dff[get_col('ad_label', l)], name=get_col('ad_label', l)) for l in range(1, 4)]
    #    return {"data": trace, "layout": go.Layout(height=600, title="Total Anomaly Days")}
#
    #return app.run_server(debug=False, port=8050, host='0.0.0.0')# app.run_server(debug=False, port=port, host=host)
    return app


if __name__ == '__main__':
    create_dashboard()