from setuptools import setup

requirements = open('requirements.txt').read().splitlines()

setup(name='mf_axa',
      description='Multiple Form AXA Handler',
      version='0.1.0',

      packages=['mf_axa', 'mf_axa.clf', 'mf_axa.post_process','mf_axa.clf.kan.form_type_clf'],
      install_requires=requirements,
      include_package_data=True,
      zip_safe=False)
