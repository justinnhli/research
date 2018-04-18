#!/usr/bin/env python3

from json import loads as read_json, dumps as json_pretty_print
from urllib.parse import urljoin as join_url

import requests


class CKANAccess:

    def __init__(self, base_url):
        self.base_url = base_url

    def _request(self, url):
        response = requests.get(url)
        if response.status_code == 200:
            return read_json(response.text)
        return None

    def list_packages(self):
        return self._request(join_url(self.base_url, 'action/package_list'))

    def list_groups(self):
        return self._request(join_url(self.base_url, 'action/group_list'))

    def list_tag(self):
        return self._request(join_url(self.base_url, 'action/tag_list'))

    def get_package(self, package_id):
        return self._request(join_url(self.base_url, 'action/package_show?id={}'.format(package_id)))

    def get_tag(self, tag_id):
        return self._request(join_url(self.base_url, 'action/tag_show?id={}'.format(tag_id)))

    def get_group(self, group_id):
        return self._request(join_url(self.base_url, 'action/package_show?id={}'.format(package_id)))

    def search_packages(self, terms, limit=10):
        json = self._request(join_url(self.base_url, 'action/package_search?q={terms}&rows={limit}'.format(**locals())))
        if json is not None and json['result']['results']:
            return [CKANDataset(self, result_json) for result_json in json['result']['results']]
        return None

    # http://demo.ckan.org/api/3/action/resource_search?query=name:District%20Names


class CKANDataset:

    def __init__(self, ckan_access, ckan_json):
        self.ckan = ckan_access
        self.json = ckan_json

    @property
    def id(self):
        return self.get_property('id')

    @property
    def author(self):
        return self.get_property('author')

    @property
    def maintainer(self):
        return self.get_property('maintainer')

    @property
    def title(self):
        return self.get_property('title')

    @property
    def name(self):
        return self.get_property('name')

    @property
    def url(self):
        return self.get_property('url')

    @property
    def formats(self):
        resources = self.get_property('resources')
        if resources is None:
            return None
        result = []
        for resource in resources:
            if 'format' in resource:
                result.append(resource['format'])
        return result

    def get_property(self, prop):
        if prop in self.json:
            if self.json[prop] != '':
                return self.json[prop]
        return None

    def get_resource_url(self, resource_format):
        resources = self.get_property('resources')
        if resources is None:
            return None
        for resource in resources:
            if resource.get('format') == resource_format and 'url' in resource:
                return resource['url']
        return None

    def pretty_print(self):
        print(json_pretty_print(self.json), sort_keys=True, indent=4)

    def get_dataset(self):
        return self.ckan.get_package(self.id)


def main():
    demo_root = 'http://catalog.data.gov/api/3/'
    ckan = CKANAccess(demo_root)
    #results = ckan.search_packages('wake county colleges universities')
    results = ckan.search_packages('rdf demographics')
    if results is not None:
        for result in results:
            print(result.title)
            print(result.name)
            print(result.formats)
            print(result.get_resource_url('RDF'))
            print()


if __name__ == '__main__':
    main()
