const { db } = require("../db/index");
const util = require("util");
const { spawn } = require('child_process');
const fs = require('fs').promises;
const path = require('path');
const csv = require('csv-stringify');
const csvStringify = util.promisify(csv.stringify);

// Convert db.query into a promise-based function
const queryAsync = util.promisify(db.query).bind(db);

// Error handler utility
const errorHandler = (error, res) => {
    console.error('Error details:', error);
    res.status(error.status || 500).json({
        message: error.message || "Internal server error",
        error: error.details || error,
    });
};

// Function to find Python executable
async function findPythonExecutable() {
    const pythonCommands = ['python3', 'python', 'py'];
    
    for (const cmd of pythonCommands) {
        try {
            const process = spawn(cmd, ['--version']);
            const result = await new Promise((resolve, reject) => {
                process.on('error', reject);
                process.on('close', code => resolve(code === 0));
            });
            
            if (result) {
                console.log(`Found Python executable: ${cmd}`);
                return cmd;
            }
        } catch (err) {
            console.log(`Command ${cmd} not available`);
            continue;
        }
    }
    
    throw new Error('No Python executable found. Please ensure Python is installed and in PATH');
}

// Function to convert form data to DataFrame format
async function convertToDataFrame(formData) {
    // Helper function to determine if a value needs quotes
    function needsQuotes(value) {
        return value && (
            value.includes(',') || 
            value.includes('"') ||
            value.includes('\n') ||
            value.includes('\r') ||
            value.length > 30  // Quote longer text fields
        );
    }

    // Helper function to format a value for CSV
    function formatCSVValue(value) {
        if (!value) return '';
        
        // If the value needs quotes, wrap it and escape existing quotes
        if (needsQuotes(value)) {
            return `"${value.replace(/"/g, '""')}"`;
        }
        
        return value;
    }

    console.log(formData.email)

    // Create a single row of data matching training data format
    const row = {
        "Username": formData.email,
        "Full Name:": formData.fullname,
        "Age:": formData.age,
        "Gender:": formData.gender,
        "Civil Status:": formData.civil_status,
        "1. Are you employed?": formData.employment_status,
        "If yes, please describe your employment situation (e.g., full-time, part-time, remote, etc.) and how it supports your ability to care for a pet. If your answer is no, please indicate 'None'.": formData.employment_status_reason,
        "2. Are you genuinely interested in adopting a pet?": formData.interest_in_adoption,
        "Please explain why you want to adopt a pet, or why you do not want to adopt one.": formData.interest_in_adoption_reason,
        "3. Do you own your home or rent?": formData.housing_ownership,
        "If renting, do you have permission from your landlord to have pets? If your answer is own, please indicate 'None'.": formData.housing_ownership_reason,
        "4. Is your area gated?": formData.gated_community_status,
        "If your answer is no, how can we ensure the safety of the pet? If your answer is yes, please indicate 'None'.": formData.gated_community_status_reason,
        "5. Will your pet primarily stay indoors or outdoors?": formData.pet_living_environment,
        "If outdoors, please describe the outdoor setup, including the size of the area, type of shelter provided, and how you will ensure the pet's safety and comfort outdoors. If your answer is indoors, please indicate 'None'.": formData.pet_living_environment_reason,
        "6. Do you have children in your household?": formData.children_in_household,
        "If yes, please list their ages and how they interact with pets. If your answer is no, please indicate 'None'.": formData.children_in_household_reason,
        "7. Do any household members have allergies to pets?": formData.household_pet_allergies,
        "If yes, how do you manage these allergies? If your answer is no, please indicate 'None'.": formData.household_pet_allergies_reason,
        "8. Is your family prepared to welcome a new family member and commit to their lifelong care?": formData.family_commitment_to_pet,
        "If yes, please explain how you will take care of the pet. If your answer is no, please explain how you will still take care of the pet.": formData.family_commitment_to_pet_reason,
        "1. What species of pet have you adopted before? If you have no previous pet adoption experience, please choose 'None'.": formData.previous_pet_species,
        "Please describe your experience with this pet. If you have no previous pet adoption experience, please indicate 'None'.": formData.previous_pet_species_details,
        "2. What was the age category of your pet at the time of adoption? If you have no previous pet adoption experience, please choose 'None'.": formData.previous_pet_age_category,
        "Why did you choose this age when adopting your previous pet(s)? If you have no previous pet adoption experience, please indicate 'None'.": formData.previous_pet_age_category_details,
        "3. What was the gender of the pet you have adopted? If you have no previous pet adoption experience, please choose 'None'.": formData.previous_pet_gender,
        "Did the gender influence your choice? Why or why not? If you have no previous pet adoption experience, please indicate 'None'.": formData.previous_pet_gender_details,
        "4. What is the story behind your adopted pet? If you have no previous pet adoption experience, please choose 'None'.": formData.previous_pet_history,
        "Please share more details about the adoption/rescue story. If you have no previous pet adoption experience, please indicate 'None'.": formData.previous_pet_history_details,
        "5. What was the spay/neuter status of your pet at the time of adoption? If you have no previous pet adoption experience, please choose 'None'.": formData.previous_pet_spay_neuter_status,
        "Why was this important to you? If you have no previous pet adoption experience, please indicate 'None'.": formData.previous_pet_spay_neuter_status_details,
        "6. What was the vaccination status of your pet at the time of adoption? If you have no previous pet adoption experience, please choose 'None'.": formData.previous_pet_vaccinations_status,
        "How did you handle their vaccination needs? If you have no previous pet adoption experience, please indicate 'None'.": formData.previous_pet_vaccinations_status_details,
        "7. What was the temperament of your adopted pet? If you have no previous pet adoption experience, please choose 'None'.": formData.previous_pet_temperament,
        "How did you manage their temperament? If you have no previous pet adoption experience, please indicate 'None'.": formData.previous_pet_temperament_details,
        "1. Vaccinations are highly encouraged for the survival of cats and dogs. Are you willing to provide all needed vaccinations for your new pet?": formData.new_pet_vaccination_willingness,
        "If yes, please describe your plan for ensuring your new pet receives all necessary vaccinations. If no, please explain why.": formData.new_pet_vaccination_willingness_description,
        "2. Are you willing to sponsor the spay and neuter of your adopted pet?": formData.spay_neuter_willingness,
        "If yes, please describe your plan for arranging and financing the spay and neuter procedure for your new pet. If no, please explain why.": formData.spay_neuter_willingness_description,
        "3. If you transfer to a new house/new location, will you bring the pet with you?": formData.pet_transfer_willingness,
        "If no, please explain under what circumstances you might not bring the pet with you. If your answer is yes, please indicate 'None.'": formData.pet_transfer_willingness_description,
        "4. Are you prepared for the long-term commitment of pet ownership, which can extend beyond 10-20 years?": formData.new_pet_long_term_commitment,
        "If yes, please explain why you are ready for this commitment and how you plan to provide for the pet's needs over its lifetime. If no, please explain why.": formData.new_pet_long_term_commitment_description,
        "5. We will contact you periodically to get updates on the pet. Is this okay with you?": formData.follow_up_permission,
        "If no, please specify the reason why. If your answer is yes, please indicate 'None.'": formData.follow_up_permission_description
    };

    // Create CSV header and data rows
    const headers = Object.keys(row).map(header => needsQuotes(header) ? `"${header}"` : header).join(',');
    const values = Object.values(row).map(value => formatCSVValue(value)).join(',');
    const csvData = `${headers}\n${values}`;

    // Create temp directory if it doesn't exist
    const tempDir = path.join(__dirname, '../temp');
    await fs.mkdir(tempDir, { recursive: true });

    // Write to temporary CSV file
    const tempFilePath = path.join(tempDir, `form_${Date.now()}.csv`);
    await fs.writeFile(tempFilePath, csvData);
    console.log('CSV file created:', tempFilePath);
    
    return tempFilePath;
}

// Function to run Python script and get prediction
async function getPrediction(csvPath) {
    return new Promise(async (resolve, reject) => {
        try {
            const pythonCommand = await findPythonExecutable();
            const pythonScriptPath = path.join(__dirname, '../screening_model/screening_model/preprocess_predict.py');
            const modelDir = path.dirname(pythonScriptPath);

            console.log('Starting ML prediction with:');
            console.log('- Python script:', pythonScriptPath);
            console.log('- Model directory:', modelDir);
            console.log('- Input CSV:', csvPath);

            // Verify all required files exist
            await Promise.all([
                fs.access(pythonScriptPath),
                fs.access(path.join(modelDir, 'tfidf_vectorizer.joblib')),
                fs.access(path.join(modelDir, 'logistic_regression_model.joblib')),
                fs.access(path.join(modelDir, 'modeling_columns.json'))
            ]);

            const pythonProcess = spawn(pythonCommand, [pythonScriptPath, csvPath], {
                env: {
                    ...process.env,
                    MODEL_DIR: modelDir,
                    PYTHONUNBUFFERED: '1' // Ensure Python output isn't buffered
                }
            });

            let stdoutData = '';
            let stderrData = '';

            pythonProcess.stdout.on('data', (data) => {
                const output = data.toString();
                stdoutData += output;
                console.log('Python stdout:', output);
            });

            pythonProcess.stderr.on('data', (data) => {
                const output = data.toString();
                stderrData += output;
                console.error('Python stderr:', output);
            });

            pythonProcess.on('error', (err) => {
                console.error('Failed to start Python process:', err);
                reject(new Error(`Failed to start Python process: ${err.message}`));
            });

            pythonProcess.on('close', async (code) => {
                console.log('Python process exited with code:', code);
                console.log('Complete stdout:', stdoutData);
                console.log('Complete stderr:', stderrData);

                // Clean up temporary CSV file
                try {
                    await fs.unlink(csvPath);
                    console.log('Temporary CSV file deleted');
                } catch (err) {
                    console.error('Error deleting temporary file:', err);
                }

                if (code !== 0) {
                    reject(new Error(`ML prediction failed (exit code ${code}): ${stderrData}`));
                    return;
                }

                try {
                    // Check if the output contains valid JSON
                    const jsonMatch = stdoutData.match(/\{[\s\S]*\}/);
                
                    if (jsonMatch) {
                        // Parse and handle JSON output if present
                        const prediction = JSON.parse(jsonMatch[0]);
                
                        if (prediction.error) {
                            reject(new Error(`ML prediction error: ${prediction.error}`));
                            return;
                        }
                
                        if (!prediction.outcome || !prediction.confidence) {
                            reject(new Error('ML prediction missing required fields'));
                            return;
                        }
                
                        console.log('ML Prediction result:', prediction);
                        resolve(prediction);
                    } else {
                        // Handle non-JSON output (raw text)
                        const outcomeMatch = stdoutData.match(/Outcome\s*=\s*([\w\s]+)/);
                        const confidenceMatch = stdoutData.match(/Confidence\s*=\s*([\d.]+)/);
                
                        if (!outcomeMatch || !confidenceMatch) {
                            throw new Error('Unable to extract prediction details from Python output');
                        }
                
                        const prediction = {
                            outcome: outcomeMatch[1].trim(),
                            confidence: parseFloat(confidenceMatch[1])
                        };
                
                        console.log('ML Prediction result:', prediction);
                        resolve(prediction);
                    }
                } catch (e) {
                    reject(new Error(`Failed to parse ML prediction output: ${e.message}\nRaw output: ${stdoutData}`));
                }
            });
        } catch (error) {
            console.error('Error in getPrediction:', error);
            
            // Clean up CSV file if there's an error
            try {
                await fs.unlink(csvPath);
                console.log('Temporary CSV file deleted after error');
            } catch (err) {
                console.error('Error deleting temporary file:', err);
            }
            
            reject(error);
        }
    });
}

// Create Form Controller
exports.createForm = async (req, res) => {
    console.log('Form submission started');
    const formData = req.body;
    let csvPath;

    // Validate required fields
    if (!formData.email || !formData.fullname || !formData.age || !formData.gender || !formData.civil_status) {
        return errorHandler(
            { status: 400, message: "All required fields must be filled." },
            res
        );
    }

    db.beginTransaction(async (transactionError) => {
        if (transactionError) {
            return errorHandler(
                { status: 500, message: "Failed to start database transaction." },
                res
            );
        }

        try {
            // Convert form data to CSV and get ML prediction
            csvPath = await convertToDataFrame(formData);
            const prediction = await getPrediction(csvPath);
            const { outcome, confidence } = prediction;

            console.log('ML Prediction completed:', { outcome, confidence });

            // Insert into `adopter`
            const adopterInsertQuery = `
                INSERT INTO adopter (email) 
                VALUES (?)
            `;
            const adopterResult = await queryAsync(adopterInsertQuery, [
                formData.email,
                outcome[0],
                confidence[0]
            ]);
            const adopterId = adopterResult.insertId;

            // Insert into `personal_infos`
            const personalInfoQuery = `
                INSERT INTO personal_infos (fullname, age, gender, civil_status, adopter_id) 
                VALUES (?, ?, ?, ?, ?)`;
            const personalInfoResult = await queryAsync(personalInfoQuery, [
                formData.fullname,
                formData.age,
                formData.gender,
                formData.civil_status,
                adopterId,
            ]);
            const personalInfoId = personalInfoResult.insertId;

            // Insert into `living_situations`
            const livingSituationsQuery = `
                INSERT INTO living_situations (
                    employment_status, employment_status_reason,
                    interest_in_adoption, interest_in_adoption_reason,
                    housing_ownership, housing_ownership_reason,
                    gated_community_status, gated_community_status_reason,
                    pet_living_environment, pet_living_environment_reason,
                    children_in_household, children_in_household_reason,
                    household_pet_allergies, household_pet_allergies_reason,
                    family_commitment_to_pet, family_commitment_to_pet_reason,
                    adopter_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`;
            const livingSituationsResult = await queryAsync(livingSituationsQuery, [
                formData.employment_status,
                formData.employment_status_reason,
                formData.interest_in_adoption,
                formData.interest_in_adoption_reason,
                formData.housing_ownership,
                formData.housing_ownership_reason,
                formData.gated_community_status,
                formData.gated_community_status_reason,
                formData.pet_living_environment,
                formData.pet_living_environment_reason,
                formData.children_in_household,
                formData.children_in_household_reason,
                formData.household_pet_allergies,
                formData.household_pet_allergies_reason,
                formData.family_commitment_to_pet,
                formData.family_commitment_to_pet_reason,
                adopterId,
            ]);
            const livingSituationId = livingSituationsResult.insertId;

            // Insert into `prev_pet_adoption_experiences`
            const prevPetQuery = `
                INSERT INTO prev_pet_adoption_experiences (
                    previous_pet_species, previous_pet_species_details,
                    previous_pet_age_category, previous_pet_age_category_details,
                    previous_pet_gender, previous_pet_gender_details,
                    previous_pet_history, previous_pet_history_details,
                    previous_pet_spay_neuter_status, previous_pet_spay_neuter_status_details,
                    previous_pet_vaccinations_status, previous_pet_vaccinations_status_details,
                    previous_pet_temperament, previous_pet_temperament_details,
                    adopter_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`;
            const prevPetResult = await queryAsync(prevPetQuery, [
                formData.previous_pet_species,
                formData.previous_pet_species_details,
                formData.previous_pet_age_category,
                formData.previous_pet_age_category_details,
                formData.previous_pet_gender,
                formData.previous_pet_gender_details,
                formData.previous_pet_history,
                formData.previous_pet_history_details,
                formData.previous_pet_spay_neuter_status,
                formData.previous_pet_spay_neuter_status_details,
                formData.previous_pet_vaccinations_status,
                formData.previous_pet_vaccinations_status_details,
                formData.previous_pet_temperament,
                formData.previous_pet_temperament_details,
                adopterId,
            ]);
            const prevPetId = prevPetResult.insertId;

            // Insert into `commitment_to_pet_ownerships`
            const commitmentQuery = `
                INSERT INTO commitment_to_pet_ownerships (
                    new_pet_vaccination_willingness, new_pet_vaccination_willingness_description,
                    spay_neuter_willingness, spay_neuter_willingness_description,
                    pet_transfer_willingness, pet_transfer_willingness_description,
                    new_pet_long_term_commitment, new_pet_long_term_commitment_description,
                    follow_up_permission, follow_up_permission_description,
                    adopter_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`;
            const commitmentResult = await queryAsync(commitmentQuery, [
                formData.new_pet_vaccination_willingness,
                formData.new_pet_vaccination_willingness_description,
                formData.spay_neuter_willingness,
                formData.spay_neuter_willingness_description,
                formData.pet_transfer_willingness,
                formData.pet_transfer_willingness_description,
                formData.new_pet_long_term_commitment,
                formData.new_pet_long_term_commitment_description,
                formData.follow_up_permission,
                formData.follow_up_permission_description,
                adopterId,
            ]);
            const commitmentId = commitmentResult.insertId;

            if (outcome[0] === "A") {
                // Insert into `screenings`
                const screeningQuery = `
                    INSERT INTO screenings (
                        adopter_id, personal_info_id, living_situation_id, 
                        prev_pet_adoption_experience_id, commitment_to_pet_ownership_id, 
                        status,
                        confidence
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)`;
                await queryAsync(screeningQuery, [
                    adopterId,
                    personalInfoId,
                    livingSituationId,
                    prevPetId,
                    commitmentId,
                    "Approve",
                    confidence
                ]);

                // Commit transaction
                db.commit((commitError) => {
                    if (commitError) {
                        throw { status: 500, message: "Error committing transaction." };
                    }
                    
                    res.status(201).json({ 
                        message: "Form successfully submitted.sd",
                        prediction: outcome[0],
                        confidence: confidence[0],
                        adopter_id: adopterId
                    });
                });
            } else {
                // Insert into `screenings`
                const screeningQuery = `
                    INSERT INTO screenings (
                        adopter_id, personal_info_id, living_situation_id, 
                        prev_pet_adoption_experience_id, commitment_to_pet_ownership_id, 
                        status,
                        confidence
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)`;
                await queryAsync(screeningQuery, [
                    adopterId,
                    personalInfoId,
                    livingSituationId,
                    prevPetId,
                    commitmentId,
                    "Disapprove",
                    confidence
                ]);

                // Commit transaction
                db.commit((commitError) => {
                    if (commitError) {
                        throw { status: 500, message: "Error committing transaction." };
                    }
                    
                    res.status(201).json({ 
                        message: "Form successfully submitted",
                        prediction: outcome[0],
                        confidence: confidence[0]
                    });
                });
            }
        } catch (error) {
            console.error('Error in transaction:', error);
            
            // Cleanup temp file if it exists
            if (csvPath) {
                try {
                    await fs.unlink(csvPath);
                    console.log('Temporary CSV file deleted after transaction error:', csvPath);
                } catch (err) {
                    console.error('Error deleting temporary file:', err);
                }
            }
            
            db.rollback(() => errorHandler({
                status: 500,
                message: "Error processing form submission",
                details: error.message
            }, res));
        }
    });
};