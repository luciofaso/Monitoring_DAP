# -*- coding: utf-8 -*-
__author__ = 'Luciano Raso'
__copyright__ = 'Copyright 2018'
__license__ = 'GNU GPL'

import pandas as pd
import numpy as np
from xml.etree.ElementTree import Element as XmlNode
from xml.etree.ElementTree import SubElement, tostring, ElementTree
from xml.dom import minidom
from dicttoxml import dicttoxml





def get_policy_def(results_df: pd.DataFrame, actions_name: list, posit_policy_name: int = None):
    """Find policies definition from results table

    Args:
        results_df:
        actions_name:
        posit_policy_name:

    Returns:
        policy_def (dict): policies definition
    """

    policy_table = results_df[['policy'] + actions_name].drop_duplicates(keep='first')

    policies_def = {}
    for policy in policy_table['policy']:
        actions_value = policy_table[policy_table['policy'] == policy][actions_name].values[0]

        policies_def[policy] = dict(zip(actions_name, actions_value))

    return policies_def


def find_precedents(policy: str, policies_def: dict, actions_order: dict):
    """Identify policies that are direct precedents (i.e. 1 action order) to a given policy

    Args:
        policy:
        policies_def: policy definition. Dictionary (policies) of dictionaries (actions).
        actions_order: order of actions (scalable actions). Dictionary of list.

    Returns:
        precedents: list of policies directly precedent to the given policy
    """

    precedents = []

    for pol in policies_def:
        difference = [0] * len(actions_order)
        for index_action, action in enumerate(actions_order):
            difference[index_action] = actions_order[action].index(policies_def[policy][action]) - \
                                       actions_order[action].index(policies_def[pol][action])

        if difference.count(0) == (len(actions_order) - 1) and difference.count(1) == 1: # include direct precedent
            precedents.append(pol)

    return (precedents)


def find_pathway(policy: str, policies_def: dict, actions_order: dict):
    """Find the entire pathway, made of all precedents policies and the policy itself, of a given policy

    Args:
        policy:
        policies_def:
        actions_order:

    Returns:
        pathway : list of policies
    """

    pathway = [policy]

    precedents = find_precedents(policy, policies_def, actions_order)
    for precedent in precedents:
        pathway.extend( find_pathway(precedent, policies_def, actions_order) )

    return pathway





def create_patwhays_map(elements: list = [], scenarios: pd.DataFrame = None, graph_settings: dict = {},
                        condition_based_pathway = True, prettified = True):
    """create .pathway file to be read by the Deltares Pathway Generator

    Args:
        elements: list of actions and pathways. First action is the current situation. Each action is defined by dict.
        pathways are actions that have the additional attribute 'combined_to', wich identify the action's name to which
        the action defined in caption must be combined to.
            required attributes:
                caption (str): action's name
                tippingpointvalue (float): final condition, termination of action
                combined_to: required only for pathways.
            semi-optional attributes
                predecessor:  -1 if "Current Situation", node 1
                color: xxx, black if not defined
            optional attributes: default = 0
                tweakx
                initialcosts
                transfercosts
                recurrentcosts
                cobenefits
                score1
                score2
                score3
                dimmed

        scenarios:

        graph_settings: dictionary that define the graph settings.
            required arguments:
                maxtippingpoint: the lower boundary of the scale
            optional arguments:
                mintippingpoint: the upper boundary of the scale, 0 by default
                xaxistitle (str): the caption of the scale, [No Caption] by default
                ticksxaxis: number of ticks in the scale
                leftmargin:
                titlesareawidth:
                topmargin:
                bottommargin:
                rightmargin:
                yoffset:
                overshoot:
                showxaxis:
                ticksxaxis:
                xaxistitle:
                drawingareawidth:
                drawingareaheight:
                fontsize:
                xaxisdecimals:

        condition_based_pathway:

        prettified:

    Returns:
        xml_pathway_map: pathway xml file/string, to be loaded in the Pathway Generator
    """

    def add_xml_pathway_features(elements):
        """ add features to the dictionary

        """

        actions = [element for element in elements if 'combined_to' not in element]

        # pathways = [element for element in elements if 'combined to' in element]
        set_ycoord(actions)

        for i,element in enumerate(elements):

            element['id'] = str(i+1)
            element['predecessorid'] = get_id(element['predecessor'], actions) if 'predecessor' in element else str(-1)

            if 'combined_to' in element: # is an pathway
                y_coord = [el['ycoord'] for el in actions if el['caption']==element['combined_to']][0]
                element['ycoord'] = str(int(y_coord)+10)
                element['caption'] = element['caption'] + ' + ' + element['combined_to']
                element['combinationsecond'] = str( get_id( element['combined_to'] , actions ) )
                element['combinationtype'] = 'ctCombine'
            else: # is an action
                element['combinationsecond'] = str(-1) # actions have not combination second, only pathways have it
                element['combinationtype'] = 'ctNone'
            # other features not necessary are not included
            # action['tweakx'] = str(0)


    def get_id(action_name: str, actions: list, key_name = 'caption') -> str:
        """find the id from the action name from a list of actions  """

        id = [action['id'] for action in actions if action[key_name]==action_name][0]
        assert isinstance(id, str), 'action id is not a string'
        return id


    def set_ycoord(actions, height_plot = 600, lower_margin = 100):
        """set the y coord, depending on the number of actions"""

        n_actions = len(actions)
        spacing = (height_plot - lower_margin) / n_actions
        for i, action in enumerate(actions):
            action['ycoord'] = str(int(spacing * (len(actions)-i)))


    def dict_to_xml(tag: str, d: dict, attribute:dict = {}):
        '''
        Turn a simple dict of key/value pairs into XML
        '''
        elem = XmlNode(tag,attribute)
        for key, val in d.items():
            child = XmlNode(key)
            child.text = str(val)
            elem.append(child)
        return elem


    def prettify(elem):
        """Return a pretty-printed XML string for the Element"""

        rough_string = tostring(elem, 'utf-8')
        reparsed = minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent="  ")


    if elements == []:
        elements.append({'caption':'current situation'})

    if scenarios == None: # no scenarios provided, set the default case
        # default case
        begin_year = 2018
        end_year = 2100# get them from scenarios
    else:
        begin_year = scenarios.iloc[:,0].index.year
        end_year = scenarios.iloc[:,0].index.year

    # pathway
    xml_pathway = XmlNode('pathways')
    xml_pathwaytype = SubElement(xml_pathway,'pathwaytype')
    xml_pathwaytype.text = 'conditionbased' if condition_based_pathway == True else 'timebased'
    xml_begin_year = SubElement(xml_pathway,'BeginYear')
    xml_begin_year.text = str(begin_year)
    xml_end_year = SubElement(xml_pathway,'EndYear')
    xml_end_year.text = str(end_year)
    xml_current_element = SubElement(xml_pathway,'CurrentElement')
    xml_current_element.text = str(1)

    # pathway elements:
    # add necessary features to elements
    add_xml_pathway_features(elements)

    # add elements to pathway node
    xml_elements = []
    for element in elements:
        xml_element = dict_to_xml('element',element, {'id':element['id']})
        xml_elements.append(xml_element)

    xml_pathway.extend(xml_elements)

    # Graph settings
    xml_graph_settings = dict_to_xml('graphsettings' , graph_settings)

    # create xml string
    create_string = prettify if prettified is True else tostring

    string_pathways = create_string(xml_pathway)
    string_graph_settings = create_string(xml_graph_settings)


    return string_pathways + string_graph_settings




