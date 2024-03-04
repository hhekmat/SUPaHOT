import json
from datetime import datetime
import os

def parse_fhir_json(file_path):
    relevant_resources = []
    with open(file_path, 'r') as f:
        fhir_data = json.load(f)
        if 'entry' in fhir_data:
            for resource in fhir_data['entry']:
                if is_relevant(resource):
                    relevant_resources.append(extract_display_name_date(resource)) 
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
    type = resource['resource']['resourceType']
    if type not in relevant_resource_types:
        return False
    elif type == 'Condition':
        if resource['resource']['clinicalStatus']['coding'][0]['system'] != 'http://terminology.hl7.org/CodeSystem/condition-clinical' or resource['resource']['clinicalStatus']['coding'][0]['code'] != 'active':
            return False
    elif type == "MedicationRequest":
        if 'medicationCodeableConcept' not in resource['resource'].keys():
            return False
    return True

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
    value = ''
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
        value = str(resource['resource']['valueQuanity']['value']) + ' ' + resource['resource']['valueQuanity']['unit']
    elif type == 'Procedure':
        display_name = resource['resource']['code']['text']
        date = format_date(resource['resource']['performedPeriod']['start'])
    return (type + display_name + '-' + date + value).replace(' ', '')

if __name__ == "__main__":
    patient_data_folder = os.path.expanduser("~/Desktop/data_processing/mock_patients")
    resources_data_folder = os.path.expanduser("~/Desktop/data_processing/all_resources")
    file_names = os.listdir(patient_data_folder)
    for file_name in file_names:
        if file_name == '.DS_Store' or file_name == 'licenses':
            continue
        file_path = os.path.join(patient_data_folder, file_name)
        print(file_path)
        relevant_resources = parse_fhir_json(file_path)
        file_name = file_name[:-5] + 'resources.txt'
        resources_file_path = os.path.join(resources_data_folder, file_name)
        with open(resources_file_path, 'w') as file:
            for item in relevant_resources:
                file.write(item + '\n')
