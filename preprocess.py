import json
from datetime import datetime
import os

global_resource_dict = {}

def parse_fhir_json(file_path):
    relevant_resources = []
    resource_counter = {}
    relevant_resource_types = ['allergyIntolerance', 'Condition', 'Encounter', 'Immunization', 'MedicationRequest', 'Observation', 'Procedure']
    for rt in relevant_resource_types:
        resource_counter[rt] = 0
    global global_resource_dict
    with open(file_path, 'r') as f:
        fhir_data = json.load(f)
        if 'entry' in fhir_data:
            for resource in reversed(fhir_data['entry']):
                # print(is_relevant(resource))
                (relevance, rt) = is_relevant(resource)
                if relevance:
                    if resource_counter[rt] < 64:
                        label = extract_display_name_date(resource)
                        relevant_resources.append(label) 
                        global_resource_dict[label] = str(resource)
                        print(resource)
                        print(label)
                        resource_counter[rt] += 1
    # print('global resource dict ', global_resource_dict)
    return relevant_resources

def is_relevant(resource):
    '''
    RELEVANT RESOURCES
    allergyIntolerances
    + llmConditions (conditions that are active and have proper URL)
    + encounters.uniqueDisplayNames (remove duplicates, only keep most recent instance)
    + immunizations
    + llmMedications (outpatient/active medications)
    + observations.uniqueDisplayNames (remove duplicates, only keep most recent instance)
    + procedures.uniqueDisplayNames (remove duplicates, only keep most recent instance)
    '''
    relevant_resource_types = ['allergyIntolerance', 'Condition', 'Encounter', 'Immunization', 'MedicationRequest', 'Observation', 'Procedure']
    rt = resource['resource']['resourceType']
    if rt not in relevant_resource_types:
        return (False, rt)
    elif rt == 'Condition':
        if resource['resource']['clinicalStatus']['coding'][0]['system'] != 'http://terminology.hl7.org/CodeSystem/condition-clinical' or resource['resource']['clinicalStatus']['coding'][0]['code'] != 'active':
            return (False, rt)
    elif rt == "MedicationRequest":
        if 'medicationCodeableConcept' not in resource['resource'].keys():
            return (False, rt)
    return (True, rt)

def extract_display_name_date(resource):
    '''
    SWIFT CODE FOR FINAL
    var functionCallIdentifier: String {
        resourceType.filter { !$0.isWhitespace }
            + displayName.filter { !$0.isWhitespace }
            + "-"
            + (date.map { FHIRResource.dateFormatter.string(from: $0) } ?? "")
    }
    '''
    def format_date(date):
        input_date = datetime.strptime(date, "%Y-%m-%dT%H:%M:%S%z")
        date = input_date.strftime("%m-%d-%Y")
        return date
    type = resource['resource']['resourceType']
    if type in ['allergyIntolerance', 'Condition']:
        display_name = resource['resource']['code']['text']
        date = format_date(resource['resource']['recordedDate'])
    elif type == 'Encounter':
        display_name = resource['resource']['type'][0]['text']
        date = format_date(resource['resource']['period']['start'])
    elif type == 'Immunization':
        display_name = resource['resource']['vaccineCode']['text']
        date = format_date(resource['resource']['occurrenceDateTime'])
    elif type == 'MedicationRequest':
        display_name = resource['resource']['medicationCodeableConcept']['text']
        date = format_date(resource['resource']['authoredOn'])
    elif type == 'Observation':
        display_name = resource['resource']['code']['text']
        date = format_date(resource['resource']['effectiveDateTime'])
    elif type == 'Procedure':
        display_name = resource['resource']['code']['text']
        date = format_date(resource['resource']['performedPeriod']['start'])
    return (type + ' ' + display_name + ' ' + date)

def populate_global_resources(patient_data_folder):
    global global_resource_dict
    resources_data_folder = "./all_resources"
    file_names = os.listdir(patient_data_folder)
    
    for file_name in file_names:
        if file_name in ('.DS_Store', 'licenses'):
            continue
        file_path = os.path.join(patient_data_folder, file_name)
        relevant_resources = parse_fhir_json(file_path)
        
        resources_file_name = file_name[:-5] + 'resources.txt'
        resources_file_path = os.path.join(resources_data_folder, resources_file_name)
        
        with open(resources_file_path, 'w') as file:
            for item in relevant_resources:
                file.write(item + '\n')

if __name__ == "__main__":
    patient_data_folder = "./mock_patients"
    resources_data_folder = "./all_resources"
    populate_global_resources(patient_data_folder)
    file_names = os.listdir(patient_data_folder)
    for file_name in file_names:
        if file_name == '.DS_Store' or file_name == 'licenses':
            continue
        file_path = os.path.join(patient_data_folder, file_name)
        # print(file_path)
        relevant_resources = parse_fhir_json(file_path)
        file_name = file_name[:-5] + 'resources.txt'
        resources_file_path = os.path.join(resources_data_folder, file_name)
        with open(resources_file_path, 'w') as file:
            for item in relevant_resources:
                file.write(item + '\n')
