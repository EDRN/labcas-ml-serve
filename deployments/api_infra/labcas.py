import re
import datetime
import json
import shutil
import requests
import sys
import os
sys.path.insert(0, (os.path.dirname(os.path.abspath(__file__))))
from deployments.api_infra.infra import solr_url, dummy_data, LabCAS_dataset_path, LabCAS_archive, real_LabCAS_archive
import traceback
# =============== Below function are taken from labcas_publish repo ====================================================

class MyException(Exception):
    pass

def clean_file_name(fname):
    fname=re.sub("\?|&|%|#|\+|\\| ", '_', fname)
    return fname

def solr_push(metadata, url, mock=False):

        if type(metadata)==dict:
            metadata_json=json.dumps(metadata)
        else:
            metadata_json=metadata

        url = url + '/update/json/docs'

        if mock:
            status='successfully mock-pushed to solr'
        else:
            headers = {'Accept': 'application/json', 'Content-Type': 'application/json'}
            response=requests.post(url, data=bytes(metadata_json, 'utf-8'), headers=headers, verify=False)

            if response.status_code==400:
                raise MyException("Error while publishing to Solr: "+response.text)
            else:
                print('got this response from solr: ', response.text)
                status=response.text
        return {'solr_status': status}

def delete_by_query(key_val, url):
    url = url  + '/update?commit=true'
    metadata = {'delete': {'query': key_val[0]+':'+key_val[1]}}
    print('deleting: ', url, json.dumps(metadata, indent=4), '\n')
    headers = {'Accept': 'application/json', 'Content-Type': 'application/json'}
    response = requests.post(url, data=bytes(json.dumps(metadata), 'utf-8'), headers=headers, verify=False)
    if response.status_code == 400:
        raise MyException("Error while publishing to Solr: " + response.text)
    else:
        print('got this response from solr: ', response.text)
    return response.text

def get_file_metadata_from_labcas(id):
    labcas_metadata={}
    try:
        print('Getting metadata from Solr:', solr_url + "/files/select?indent=on&wt=json&q=id:"+id)
        r = requests.get(url=solr_url + "/files/select?indent=on&wt=json&q=id:"+id, verify=False)
        print('Solr request status code:', r.status_code)
        labcas_metadata = r.json()['response']['docs'][0] if len(r.json()['response']['docs']) > 0 else {}
        return labcas_metadata
    except:
        print('Error (suppressed): Could not get metadata from LabCAS:', traceback.format_exc())
    return labcas_metadata

# =============== Above function are taken from labcas_publish repo ====================================================

def push_to_labcas_MLOutputs_collection(task_id, target_id, permissions, filename=None, filetype='N/A', user='N/A'):

    # create minimal LabCAS metadata for publishing
    if filename is not None:
        if '.' in filename and filename.lower().split('.')[-1] in ['png', 'jpeg', 'jpg', 'dcm', 'tif', 'tiff']:
            contains_image='True'
        else:
            contains_image='False'
        relative_path = os.path.join(LabCAS_dataset_path, task_id, filename)
        labcas_node_type='files'
        labcas_metadata={
            'id': relative_path,
            'name': os.path.basename(relative_path),
            'labcasId': clean_file_name(relative_path),
            'labcasName': clean_file_name(os.path.basename(relative_path)),
            'FileId': relative_path,
            'FileType': filetype,
            'labcas_node_type': labcas_node_type,
            'FileName': os.path.basename(relative_path),
            'user': user,
            'CollectionId': os.path.dirname(LabCAS_dataset_path),
            'CollectionName': os.path.dirname(LabCAS_dataset_path),
            'DatasetId':  os.path.join(LabCAS_dataset_path, task_id),
            'DatasetName': task_id,
            'size': int(os.path.getsize(os.path.join(LabCAS_archive, relative_path))),
            'FileDownloadId': 'N/A',
            'DatasetVersion': 1,
            'contains_image': contains_image,
            'OwnerPrincipal': permissions,
            'FileVersion': 1,
            'target_id': target_id,
            'FileLocation': os.path.join(real_LabCAS_archive, os.path.join(LabCAS_dataset_path, task_id)),
            'RealFileLocation': os.path.join(real_LabCAS_archive, os.path.join(LabCAS_dataset_path, task_id)),
            'PublishDate': str(datetime.datetime.now())
        }
    else:
        relative_path=os.path.join(LabCAS_dataset_path, task_id)
        labcas_node_type = 'datasets'
        labcas_metadata = {
            'id': relative_path,
            'name': task_id,
            'labcasId': relative_path,
            'labcasName': task_id,
            'labcas_node_type': labcas_node_type,
            'CollectionId': os.path.dirname(LabCAS_dataset_path),
            'CollectionName': os.path.dirname(LabCAS_dataset_path),
            'DatasetId': relative_path,
            'DatasetName': task_id,
            'OwnerPrincipal': permissions,
            'DatasetParentId': LabCAS_dataset_path,
            'DatasetVersion': 1,
            'user': user,
            'target_id': target_id,
            'PublishDate': str(datetime.datetime.now())
        }

    file_labcas_metadata=json.dumps(labcas_metadata)
    print('Publishing:', file_labcas_metadata)
    solr_push(file_labcas_metadata, solr_url+'/'+labcas_node_type)

if __name__ == "__main__":

    # create the new collection directory in labcas if does not exist:
    os.makedirs(os.path.join(LabCAS_archive, LabCAS_dataset_path), exist_ok=True)
    shutil.copy(dummy_data, os.path.join(LabCAS_archive, LabCAS_dataset_path, os.path.basename(dummy_data)))

    # Use this to publish the MLOutputs/MLOutputs dataset in LabCAS the first time:
    delete_by_query(('CollectionId', os.path.dirname(LabCAS_dataset_path)), solr_url + '/collections')
    delete_by_query(('CollectionId', os.path.dirname(LabCAS_dataset_path)), solr_url + '/datasets')
    delete_by_query(('CollectionId', os.path.dirname(LabCAS_dataset_path)), solr_url + '/files')

    dummy_file_labcas_metadata = {
        'id': os.path.join(LabCAS_dataset_path, os.path.basename(dummy_data)),
        'name': os.path.basename(dummy_data),
        'labcasId': os.path.join(LabCAS_dataset_path, os.path.basename(dummy_data)),
        'labcasName': os.path.basename(dummy_data),
        'FileId': os.path.join(LabCAS_dataset_path, os.path.basename(dummy_data)),
        'FileType': 'PNG',
        'labcas_node_type': 'files',
        'FileName': os.path.basename(dummy_data),
        'CollectionId': os.path.dirname(LabCAS_dataset_path),
        'CollectionName': os.path.dirname(LabCAS_dataset_path),
        'DatasetId': LabCAS_dataset_path,
        'DatasetName': os.path.basename(LabCAS_dataset_path),
        'size': int(os.path.getsize(os.path.join(LabCAS_archive, LabCAS_dataset_path, os.path.basename(dummy_data)))),
        'FileDownloadId': 'N/A',
        'DatasetVersion': 1,
        'FileVersion': 1,
        'FileLocation': os.path.join(real_LabCAS_archive, LabCAS_dataset_path),
        'RealFileLocation': os.path.join(real_LabCAS_archive, LabCAS_dataset_path),
        'PublishDate': str(datetime.datetime.now()),
        'contains_image': 'True',
        'OwnerPrincipal': ["cn=All MCL,ou=groups,o=MCL"] # todo: put a default one
    }
    solr_push(dummy_file_labcas_metadata, solr_url + '/files')
    collection_labcas_metadata = {
        'id': os.path.dirname(LabCAS_dataset_path),
        'name': os.path.dirname(LabCAS_dataset_path),
        'labcasId': os.path.dirname(LabCAS_dataset_path),
        'labcasName': os.path.dirname(LabCAS_dataset_path),
        'labcas_node_type': 'collections',
        'CollectionId': os.path.dirname(LabCAS_dataset_path),
        'CollectionName': os.path.dirname(LabCAS_dataset_path),
        'PublishDate': str(datetime.datetime.now())
    }
    solr_push(collection_labcas_metadata, solr_url + '/collections')
    dataset_labcas_metadata = {
        'id': LabCAS_dataset_path,
        'name': os.path.basename(LabCAS_dataset_path),
        'labcasId': LabCAS_dataset_path,
        'labcasName': os.path.basename(LabCAS_dataset_path),
        'labcas_node_type': 'datasets',
        'CollectionId': os.path.dirname(LabCAS_dataset_path),
        'CollectionName': os.path.dirname(LabCAS_dataset_path),
        'DatasetId': LabCAS_dataset_path,
        'DatasetName': os.path.basename(LabCAS_dataset_path),
        'DatasetVersion': 1,
        'PublishDate': str(datetime.datetime.now())
    }
    solr_push(dataset_labcas_metadata, solr_url + '/datasets')
