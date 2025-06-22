#!/usr/bin/env python
import argparse
import os
from typing import OrderedDict
import yaml
import numpy as np
import pandas as pd
from datetime import datetime
from report_plotter import ReportPlotter

root_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..'))
class ReportGenerator:
    def __init__(self, target_dir):
        config_filepath = os.path.join(root_path, 'plotter', 'config', 'config.yaml')
        with open(config_filepath, 'r') as f:
            self.target_signals = yaml.load(f, Loader=yaml.FullLoader)

        target_dir_fullpath = os.path.join(root_path, 'record', target_dir)
        self.output_dir = target_dir_fullpath
        data_names_filepath = os.path.join(target_dir_fullpath, 'data_names.csv')
        data_filepath = os.path.join(target_dir_fullpath, 'data.csv')
        
        self.data = pd.read_csv(data_filepath)
        self.data_names = pd.read_csv(data_names_filepath)
        self.organize_data()

        self.data_selected = OrderedDict()
        self.report_plotter = ReportPlotter('ReportPlotter')
        self.figure_height = 600
        
    
    def organize_data(self):
        self.data_dict = dict()
        for i, row in self.data_names.iterrows():
            data_name = self.data_names.iloc[i,0]
            start_col_id = int(self.data_names.iloc[i,1])
            end_col_id = int(self.data_names.iloc[i,2])
            for j in range(start_col_id, end_col_id + 1):
                self.data_dict[f'{data_name}_{j-start_col_id}'] = self.data.iloc[:, j-1]

    def generate_report(self):
        subplot_figure = None
        plot_html_str = ""  

        output_filename = os.path.join(self.output_dir, "report.html")

        target_panel = dict(self.target_signals["target_panel"])
        for sub_panel_name, signal_names in target_panel.items():
            sub_dict = OrderedDict()
            for signal_name, plot_name in signal_names.items():
                if signal_name in self.data_dict.keys():
                    data = self.data_dict[signal_name]
                    time = self.data_dict['simTime_0']
                    sub_dict[plot_name] = (data, time)
            self.data_selected[sub_panel_name] = sub_dict


        start_timestamp = self.data_dict['simTime_0'][0]
        figure_list = []
        for sub_panel_name, sub_dict in self.data_selected.items():
            legend_list = []
            value_list = []
            time_list = []
            for signal_full_name, (data, data_time) in sub_dict.items():
                time_list.append(np.array([(x - start_timestamp) for x in data_time]))
                value_list.append(np.array(data))
                legend_list.append(signal_full_name)
            
            subplot = self.report_plotter.plot_figure_plotly(x_list = time_list, 
                                                y_list = value_list,
                                                legend_list = legend_list,
                                                x_label = 'time / s',
                                                y_label = '',
                                                title = sub_panel_name,
                                                legend_prefix = '',
                                                figure_height=self.figure_height,)
            figure_list.append(subplot)  

        subplot_figure_list = [(i + 1, 1, fig) for i, fig in enumerate(figure_list)]
        subplot_figure = self.report_plotter.append_figure_to_subplot_plotly(
            subplot_figure_list, 
            len(figure_list), 
            1, 
            template="plotly_dark", 
            subplot_fig=subplot_figure, 
            vertical_spacing=0.1
        )
        plot_html_str += self.report_plotter.get_fuel_fig_html_str({"Comparison": subplot_figure})
        html_str = self.report_plotter.generate_html_fuel_report(plot_html_str)
        with open(output_filename, 'w') as f:
            f.write(html_str)

def get_last_sorted_dir(target_path):
    entries = os.listdir(target_path)
    dirs = [d for d in entries if os.path.isdir(os.path.join(target_path, d))]
    dirs.sort()
    return dirs[-1] if dirs else None
    
def main(args):
    if(len(args.target_dir) == 0):
        target_dir_fullpath = os.path.join(root_path, 'record')
        target_dir = get_last_sorted_dir(target_dir_fullpath)
    else:
        target_dir = args.target_dir
    report_generator = ReportGenerator(target_dir)
    report_generator.generate_report()

if __name__ == '__main__':
    parser = argparse.ArgumentParser('ReportGenerator')
    parser.add_argument('--target-dir', default='', type=str)
    args = parser.parse_args()
    main(args)