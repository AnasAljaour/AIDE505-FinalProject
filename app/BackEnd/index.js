express = require('express')
bodyParser = require('body-parser')
axios = require('axios')
application = express()
application.use(bodyParser.json())

const FEATURE_NAMES = [
    "Perimeter_Worst",
    "ConcavePoints_Worst",
    "Area_Worst",
    "Radius_Worst",
    "Texture_Worst",
    "Smoothness_Worst",
    "Texture_Mean",
    "Concavity_Worst"
]


application.post('/cancer-diagnosis', (req, res) => {
    
    const features = req.body;
    
    if (!features || typeof features !== 'object') {
        return res.status(400).json({ error: 'Invalid input format' });
    }
    
    
    const missingFeatures = FEATURE_NAMES.filter(feature => !(feature in features));
    if (missingFeatures.length > 0) {
        return res.status(400).json({ error: `Missing features: ${missingFeatures.join(', ')}` });
    }

    let URL = "http://Backend-Flask:5000/predict"
    axios.post(URL, features)
        .then(response => {
            
            if (!response.data.prediction) {
                return res.status(400).send({ error: 'Invalid prediction response from Flask API' });
            }
            
            res.send({ prediction: response.data.prediction });
        })
        .catch(error => {
            
            console.error('Error calling Flask API:', error);
            res.status(500).send({ error: 'Internal server error while contacting Flask API' });
        });
});


application.listen(3000, () => {
    console.log('Server is running on port 3000');
});