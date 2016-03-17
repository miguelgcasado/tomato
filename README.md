# tomato
Turkish-Ottoman Makam (Music) Analysis TOolbox

Documentation
------
Coming soon...

Installation
-------

If you want to install tomato, it is recommended to install the package and dependencies into a virtualenv. In the terminal, do the following:

    virtualenv env
    source env/bin/activate
    python setup.py install

If you want to be able to edit files and have the changes be reflected, then install the repository like this instead:

    pip install -e .

The algorithm uses several modules in Essentia. Follow the [instructions](essentia.upf.edu/documentation/installing.html) to install the library. Then you should link the python bindings of Essentia in the virtual environment:

    ln -s /usr/local/lib/python2.7/dist-packages/essentia env/lib/python2.7/site-packages

Now you can install the rest of the dependencies:

    pip install -r requirements


Basic Usage
-------

Below you can find some basic calls for analysis using the the package.

##### Audio Analysis
```python
from tomato.audio.AudioAnalyzer import AudioAnalyzer

audio_filepath = 'path/to/audio'
makam = 'makam_name'  # the makam slug. See the documentation for possible values

audioAnalyzer = AudioAnalyzer()
features = audioAnalyzer.analyze(audio_filepath, makam=makam)

# plot the features
import pylab
audioAnalyzer.plot(features)
pylab.show()

# save features to json file
audioAnalyzer.save_features(features, 'save_filename.json')
```

##### Symbolic Analysis

##### Score-informed Audio Analysis

Authors
-------
Sertan Şentürk
contact@sertansenturk.com

Reference
-------
Thesis
