from old_tool.utils_backup.config import *

# GLOBAL VAR
progr_bar_lenght = 20


def print_log(string, print_on_screen=False, print_on_file=True):
    if print_on_screen:
        print(string)
    if print_on_file:
        with open(main_path + 'results/exec_logs/' + timeExec + ".results", 'a') as logfile:
            logfile.write(string + "\n")


def progr_bar(part, tot):
    to_print = "["
    partial = int((part/tot)*progr_bar_lenght)
    for x in range(partial):
        to_print += "="
    to_print += ">"
    for x in range(progr_bar_lenght-partial):
        to_print += "-"
    to_print += "]"
    return "{}/{} {} ".format(part, tot, to_print)


def clean_progr_bar():
    to_print = ""
    for i in range(50):
        to_print += " "
    print(to_print, end="\r")
