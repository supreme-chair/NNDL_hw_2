/**
 * Titanic Binary Classifier with TensorFlow.js
 * Shallow neural network implementation for browser-based training
 * Ready for GitHub Pages deployment (no server required)
 */

// Global variables to store data and model
let trainData = null;
let testData = null;
let trainFeatures = null;
let trainLabels = null;
let valFeatures = null;
let valLabels = null;
let testFeatures = null;
let model = null;
let trainingHistory = null;
let valPredictions = null;
let testPredictions = null;
let featureNames = [];
let includeFamilyFeatures = false;
let isTraining = false;
let currentTraining = null;

// DOM Elements
const elements = {
    // Buttons
    loadDataBtn: document.getElementById('loadDataBtn'),
    preprocessBtn: document.getElementById('preprocessBtn'),
    toggleFeaturesBtn: document.getElementById('toggleFeaturesBtn'),
    initModelBtn: document.getElementById('initModelBtn'),
    trainBtn: document.getElementById('trainBtn'),
    stopTrainBtn: document.getElementById('stopTrainBtn'),
    evaluateBtn: document.getElementById('evaluateBtn'),
    predictBtn: document.getElementById('predictBtn'),
    exportBtn: document.getElementById('exportBtn'),
    saveModelBtn: document.getElementById('saveModelBtn'),
    
    // File inputs
    trainFile: document.getElementById('trainFile'),
    testFile: document.getElementById('testFile'),
    
    // Status displays
    dataStatus: document.getElementById('dataStatus'),
    preprocessStatus: document.getElementById('preprocessStatus'),
    modelStatus: document.getElementById('modelStatus'),
    trainingStatus: document.getElementById('trainingStatus'),
    metricsStatus: document.getElementById('metricsStatus'),
    predictionStatus: document.getElementById('predictionStatus'),
    
    // Display areas
    dataPreview: document.getElementById('dataPreview'),
    featureInfo: document.getElementById('featureInfo'),
    modelSummary: document.getElementById('modelSummary'),
    trainingCharts: document.getElementById('trainingCharts'),
    rocChart: document.getElementById('rocChart'),
    confusionMatrix: document.getElementById('confusionMatrix'),
    performanceMetrics: document.getElementById('performanceMetrics'),
    predictionResults: document.getElementById('predictionResults'),
    
    // Threshold slider
    thresholdSlider: document.getElementById('thresholdSlider'),
    thresholdValue: document.getElementById('thresholdValue')
};

// Initialize event listeners
function initEventListeners() {
    // Data loading
    elements.loadDataBtn.addEventListener('click', loadData);
    
    // Preprocessing
    elements.preprocessBtn.addEventListener('click', preprocessData);
    elements.toggleFeaturesBtn.addEventListener('click', toggleFamilyFeatures);
    
    // Model
    elements.initModelBtn.addEventListener('click', initModel);
    
    // Training
    elements.trainBtn.addEventListener('click', trainModel);
    elements.stopTrainBtn.addEventListener('click', stopTraining);
    
    // Evaluation
    elements.evaluateBtn.addEventListener('click', evaluateModel);
    elements.thresholdSlider.addEventListener('input', updateThreshold);
    
    // Prediction & Export
    elements.predictBtn.addEventListener('click', predictTestData);
    elements.exportBtn.addEventListener('click', exportPredictions);
    elements.saveModelBtn.addEventListener('click', saveModel);
}

// Update status display
function updateStatus(element, message, type = 'info') {
    element.textContent = message;
    element.className = 'status';
    
    switch (type) {
        case 'loading':
            element.classList.add('loading');
            break;
        case 'success':
            element.classList.add('success');
            break;
        case 'error':
            element.classList.add('error');
            break;
        default:
            // No additional class for info
            break;
    }
}

// Clear status display
function clearStatus(element) {
    element.textContent = '';
    element.className = 'status';
}

// Parse CSV data with proper handling of commas within quotes
function parseCSV(text) {
    const rows = [];
    let currentRow = [];
    let currentField = '';
    let insideQuotes = false;
    
    for (let i = 0; i < text.length; i++) {
        const char = text[i];
        
        if (char === '"') {
            // Toggle quote mode
            insideQuotes = !insideQuotes;
        } else if (char === ',' && !insideQuotes) {
            // End of field
            currentRow.push(currentField);
            currentField = '';
        } else if (char === '\n' && !insideQuotes) {
            // End of row
            currentRow.push(currentField);
            rows.push(currentRow);
            currentRow = [];
            currentField = '';
        } else {
            // Add character to current field
            currentField += char;
        }
    }
    
    // Add the last row if any
    if (currentField !== '' || currentRow.length > 0) {
        currentRow.push(currentField);
        rows.push(currentRow);
    }
    
    return rows;
}

// Load CSV data from file inputs
async function loadData() {
    const trainFile = elements.trainFile.files[0];
    const testFile = elements.testFile.files[0];
    
    if (!trainFile) {
        alert('Please select a training CSV file (train.csv)');
        return;
    }
    
    updateStatus(elements.dataStatus, 'Loading data...', 'loading');
    
    try {
        // Load training data
        const trainText = await trainFile.text();
        const trainRows = parseCSV(trainText);
        
        if (trainRows.length < 2) {
            throw new Error('Training CSV file is empty or invalid');
        }
        
        // Extract headers and data
        const trainHeaders = trainRows[0];
        const trainDataRows = trainRows.slice(1);
        
        trainData = trainDataRows.map(row => {
            const obj = {};
            trainHeaders.forEach((header, index) => {
                obj[header] = row[index];
            });
            return obj;
        });
        
        console.log(`Training data loaded: ${trainData.length} samples`);
        
        // Load test data if provided
        if (testFile) {
            const testText = await testFile.text();
            const testRows = parseCSV(testText);
            
            if (testRows.length < 2) {
                throw new Error('Test CSV file is empty or invalid');
            }
            
            const testHeaders = testRows[0];
            const testDataRows = testRows.slice(1);
            
            testData = testDataRows.map(row => {
                const obj = {};
                testHeaders.forEach((header, index) => {
                    obj[header] = row[index];
                });
                return obj;
            });
            
            console.log(`Test data loaded: ${testData.length} samples`);
        }
        
        // Display data preview
        displayDataPreview();
        
        // Enable preprocessing button
        elements.preprocessBtn.disabled = false;
        
        updateStatus(elements.dataStatus, `Data loaded successfully! Training: ${trainData.length} samples, Test: ${testData ? testData.length : 0} samples`, 'success');
        
    } catch (error) {
        console.error('Error loading data:', error);
        updateStatus(elements.dataStatus, `Error: ${error.message}`, 'error');
    }
}

// Display a preview of the loaded data
function displayDataPreview() {
    if (!trainData || trainData.length === 0) return;
    
    let html = '<h3>Data Preview (First 10 Rows)</h3>';
    html += '<table><thead><tr>';
    
    // Table headers
    const headers = Object.keys(trainData[0]);
    headers.forEach(header => {
        html += `<th>${header}</th>`;
    });
    html += '</tr></thead><tbody>';
    
    // Table rows (first 10)
    for (let i = 0; i < Math.min(10, trainData.length); i++) {
        html += '<tr>';
        headers.forEach(header => {
            html += `<td>${trainData[i][header]}</td>`;
        });
        html += '</tr>';
    }
    
    html += '</tbody></table>';
    
    // Data statistics
    html += '<div style="margin-top: 20px;">';
    html += `<p><strong>Training Samples:</strong> ${trainData.length}</p>`;
    html += `<p><strong>Features:</strong> ${headers.length}</p>`;
    
    // Calculate survival rate
    const survivedCount = trainData.filter(row => row.Survived === '1').length;
    const survivalRate = (survivedCount / trainData.length * 100).toFixed(1);
    html += `<p><strong>Survival Rate:</strong> ${survivalRate}% (${survivedCount}/${trainData.length})</p>`;
    
    // Show missing values
    const missingInfo = calculateMissingValues(trainData);
    html += '<p><strong>Missing Values:</strong></p><ul>';
    for (const [key, value] of Object.entries(missingInfo)) {
        if (value > 0) {
            const percentage = (value / trainData.length * 100).toFixed(1);
            html += `<li>${key}: ${value} (${percentage}%)</li>`;
        }
    }
    html += '</ul></div>';
    
    // Create visualization charts
    createDataVisualizations();
    
    elements.dataPreview.innerHTML = html;
}

// Calculate missing values in the dataset
function calculateMissingValues(data) {
    const missing = {};
    const headers = Object.keys(data[0]);
    
    headers.forEach(header => {
        missing[header] = 0;
    });
    
    data.forEach(row => {
        headers.forEach(header => {
            if (row[header] === '' || row[header] === null || row[header] === undefined) {
                missing[header]++;
            }
        });
    });
    
    return missing;
}

// Create data visualization charts
function createDataVisualizations() {
    if (!trainData || trainData.length === 0) return;
    
    // Survival by Sex
    const sexCounts = {};
    const sexSurvived = {};
    
    trainData.forEach(row => {
        const sex = row.Sex || 'Unknown';
        const survived = row.Survived === '1';
        
        if (!sexCounts[sex]) {
            sexCounts[sex] = 0;
            sexSurvived[sex] = 0;
        }
        
        sexCounts[sex]++;
        if (survived) sexSurvived[sex]++;
    });
    
    const sexLabels = Object.keys(sexCounts);
    const sexSurvivalRates = sexLabels.map(sex => sexSurvived[sex] / sexCounts[sex]);
    
    // Survival by Pclass
    const pclassCounts = {};
    const pclassSurvived = {};
    
    trainData.forEach(row => {
        const pclass = row.Pclass || 'Unknown';
        const survived = row.Survived === '1';
        
        if (!pclassCounts[pclass]) {
            pclassCounts[pclass] = 0;
            pclassSurvived[pclass] = 0;
        }
        
        pclassCounts[pclass]++;
        if (survived) pclassSurvived[pclass]++;
    });
    
    const pclassLabels = Object.keys(pclassCounts).sort();
    const pclassSurvivalRates = pclassLabels.map(pclass => pclassSurvived[pclass] / pclassCounts[pclass]);
    
    // Create charts using tfjs-vis
    const surfaceSex = { name: 'Survival by Sex', tab: 'Data Analysis' };
    const surfacePclass = { name: 'Survival by Passenger Class', tab: 'Data Analysis' };
    
    // Render charts
    tfvis.render.barchart(surfaceSex, {
        values: sexSurvivalRates,
        labels: sexLabels
    }, {
        xLabel: 'Sex',
        yLabel: 'Survival Rate',
        height: 300
    });
    
    tfvis.render.barchart(surfacePclass, {
        values: pclassSurvivalRates,
        labels: pclassLabels
    }, {
        xLabel: 'Passenger Class',
        yLabel: 'Survival Rate',
        height: 300
    });
}

// Toggle family features (FamilySize and IsAlone)
function toggleFamilyFeatures() {
    includeFamilyFeatures = !includeFamilyFeatures;
    elements.toggleFeaturesBtn.textContent = `Toggle Family Features (${includeFamilyFeatures ? 'On' : 'Off'})`;
    elements.toggleFeaturesBtn.classList.toggle('success', includeFamilyFeatures);
    
    updateStatus(elements.preprocessStatus, 
        `Family features ${includeFamilyFeatures ? 'enabled' : 'disabled'}. Click "Run Preprocessing" to apply changes.`, 
        'info');
}

// Preprocess the data
function preprocessData() {
    if (!trainData || trainData.length === 0) {
        updateStatus(elements.preprocessStatus, 'No training data loaded', 'error');
        return;
    }
    
    updateStatus(elements.preprocessStatus, 'Preprocessing data...', 'loading');
    
    try {
        // Extract features and labels from training data
        const { featuresArray, labelsArray, featureNames: names } = extractFeaturesAndLabels(trainData, true);
        featureNames = names;
        
        // Convert to tensors
        const featuresTensor = tf.tensor2d(featuresArray);
        const labelsTensor = tf.tensor2d(labelsArray, [labelsArray.length, 1]);
        
        // Create training/validation split (80/20)
        const splitIndex = Math.floor(featuresArray.length * 0.8);
        
        trainFeatures = featuresTensor.slice(0, splitIndex);
        trainLabels = labelsTensor.slice(0, splitIndex);
        valFeatures = featuresTensor.slice(splitIndex);
        valLabels = labelsTensor.slice(splitIndex);
        
        console.log('Preprocessed data shapes:');
        console.log(`  Train features: ${trainFeatures.shape}`);
        console.log(`  Train labels: ${trainLabels.shape}`);
        console.log(`  Validation features: ${valFeatures.shape}`);
        console.log(`  Validation labels: ${valLabels.shape}`);
        
        // Display feature information
        displayFeatureInfo();
        
        // Enable model initialization button
        elements.initModelBtn.disabled = false;
        
        updateStatus(elements.preprocessStatus, 
            `Preprocessing complete! Features: ${featureNames.length}, Training: ${trainFeatures.shape[0]}, Validation: ${valFeatures.shape[0]}`, 
            'success');
            
    } catch (error) {
        console.error('Error preprocessing data:', error);
        updateStatus(elements.preprocessStatus, `Error: ${error.message}`, 'error');
    }
}

// Extract features and labels from data
function extractFeaturesAndLabels(data, isTraining = true) {
    const featuresArray = [];
    const labelsArray = [];
    
    // Calculate median age and mode embarked from training data
    let medianAge = 0;
    let modeEmbarked = 'S'; // Default to Southampton
    
    if (isTraining) {
        // Calculate median age (ignore missing values)
        const ages = data
            .filter(row => row.Age && row.Age !== '')
            .map(row => parseFloat(row.Age))
            .sort((a, b) => a - b);
        
        if (ages.length > 0) {
            const mid = Math.floor(ages.length / 2);
            medianAge = ages.length % 2 === 0 ? (ages[mid - 1] + ages[mid]) / 2 : ages[mid];
        }
        
        // Calculate mode embarked
        const embarkedCounts = {};
        data.forEach(row => {
            if (row.Embarked && row.Embarked !== '') {
                embarkedCounts[row.Embarked] = (embarkedCounts[row.Embarked] || 0) + 1;
            }
        });
        
        let maxCount = 0;
        for (const [key, count] of Object.entries(embarkedCounts)) {
            if (count > maxCount) {
                maxCount = count;
                modeEmbarked = key;
            }
        }
    }
    
    // Calculate mean and std for standardization (only from training data)
    let meanAge = 0, stdAge = 1, meanFare = 0, stdFare = 1;
    
    if (isTraining) {
        const validAges = data
            .filter(row => row.Age && row.Age !== '')
            .map(row => parseFloat(row.Age));
        const validFares = data
            .filter(row => row.Fare && row.Fare !== '')
            .map(row => parseFloat(row.Fare));
        
        if (validAges.length > 0) {
            meanAge = validAges.reduce((a, b) => a + b, 0) / validAges.length;
            stdAge = Math.sqrt(validAges.map(a => Math.pow(a - meanAge, 2)).reduce((a, b) => a + b, 0) / validAges.length) || 1;
        }
        
        if (validFares.length > 0) {
            meanFare = validFares.reduce((a, b) => a + b, 0) / validFares.length;
            stdFare = Math.sqrt(validFares.map(f => Math.pow(f - meanFare, 2)).reduce((a, b) => a + b, 0) / validFares.length) || 1;
        }
    }
    
    // Process each row
    data.forEach(row => {
        const features = [];
        
        // Handle missing values
        const age = row.Age && row.Age !== '' ? parseFloat(row.Age) : medianAge;
        const fare = row.Fare && row.Fare !== '' ? parseFloat(row.Fare) : 0;
        const embarked = row.Embarked && row.Embarked !== '' ? row.Embarked : modeEmbarked;
        const sex = row.Sex || 'male';
        const pclass = row.Pclass || '3';
        
        // Standardize Age and Fare
        features.push((age - meanAge) / stdAge); // Age (standardized)
        features.push((fare - meanFare) / stdFare); // Fare (standardized)
        
        // One-hot encode Sex: [male, female]
        features.push(sex === 'female' ? 1 : 0); // Female
        features.push(sex === 'male' ? 1 : 0);   // Male
        
        // One-hot encode Pclass: [1, 2, 3]
        features.push(pclass === '1' ? 1 : 0); // Pclass 1
        features.push(pclass === '2' ? 1 : 0); // Pclass 2
        features.push(pclass === '3' ? 1 : 0); // Pclass 3
        
        // One-hot encode Embarked: [C, Q, S]
        features.push(embarked === 'C' ? 1 : 0); // Cherbourg
        features.push(embarked === 'Q' ? 1 : 0); // Queenstown
        features.push(embarked === 'S' ? 1 : 0); // Southampton
        
        // SibSp and Parch (raw values)
        features.push(parseInt(row.SibSp) || 0);
        features.push(parseInt(row.Parch) || 0);
        
        // Optional family features
        if (includeFamilyFeatures) {
            const familySize = (parseInt(row.SibSp) || 0) + (parseInt(row.Parch) || 0) + 1;
            features.push(familySize); // FamilySize
            features.push(familySize === 1 ? 1 : 0); // IsAlone
        }
        
        featuresArray.push(features);
        
        // Extract label if this is training data
        if (isTraining) {
            labelsArray.push([parseInt(row.Survived) || 0]);
        }
    });
    
    // Feature names for display
    const featureNames = [
        'Age (std)', 'Fare (std)', 
        'Sex: Female', 'Sex: Male',
        'Pclass: 1', 'Pclass: 2', 'Pclass: 3',
        'Embarked: C', 'Embarked: Q', 'Embarked: S',
        'SibSp', 'Parch'
    ];
    
    if (includeFamilyFeatures) {
        featureNames.push('FamilySize', 'IsAlone');
    }
    
    return {
        featuresArray,
        labelsArray,
        featureNames,
        stats: { meanAge, stdAge, meanFare, stdFare, medianAge, modeEmbarked }
    };
}

// Display feature information
function displayFeatureInfo() {
    let html = '<h3>Feature Information</h3>';
    html += `<p><strong>Total Features:</strong> ${featureNames.length}</p>`;
    html += '<ul>';
    
    featureNames.forEach((name, index) => {
        html += `<li>${index + 1}. ${name}</li>`;
    });
    
    html += '</ul>';
    
    if (trainFeatures) {
        html += `<p><strong>Training Samples:</strong> ${trainFeatures.shape[0]}</p>`;
        html += `<p><strong>Validation Samples:</strong> ${valFeatures.shape[0]}</p>`;
    }
    
    elements.featureInfo.innerHTML = html;
}

// Initialize the neural network model
function initModel() {
    updateStatus(elements.modelStatus, 'Initializing model...', 'loading');
    
    try {
        // Create a sequential model
        model = tf.sequential();
        
        // Add a single hidden layer with 16 neurons and ReLU activation
        model.add(tf.layers.dense({
            units: 16,
            activation: 'relu',
            inputShape: [featureNames.length]
        }));
        
        // Add output layer with 1 neuron and sigmoid activation for binary classification
        model.add(tf.layers.dense({
            units: 1,
            activation: 'sigmoid'
        }));
        
        // Compile the model
        model.compile({
            optimizer: tf.train.adam(0.001),
            loss: 'binaryCrossentropy',
            metrics: ['accuracy']
        });
        
        // Display model summary
        displayModelSummary();
        
        // Enable training button
        elements.trainBtn.disabled = false;
        
        updateStatus(elements.modelStatus, 'Model initialized successfully!', 'success');
        
    } catch (error) {
        console.error('Error initializing model:', error);
        updateStatus(elements.modelStatus, `Error: ${error.message}`, 'error');
    }
}

// Display model summary
function displayModel
