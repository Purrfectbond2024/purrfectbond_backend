const { db } = require("../db/index");
const util = require("util");
const fs = require("fs").promises;
const path = require("path");
const csv = require("csv-stringify");
const { spawn } = require("child_process");

// Convert db.query to promise-based
const queryAsync = util.promisify(db.query).bind(db);
const TEMP_DIR = path.join(__dirname, "../temp");
const PYTHON_TIMEOUT = 30000;

// Predefined headers for screening and recommendation data
const SCREENING_HEADERS = [
    "Timestamp", "Username", "Full Name:", "Age:", "Gender:", "Civil Status:",
    "1. Are you employed?",
    "If yes, please describe your employment situation (e.g., full-time, part-time, remote, etc.) and how it supports your ability to care for a pet. If your answer is no, please indicate 'None'.",
    "2. Are you genuinely interested in adopting a pet?",
    "Please explain why you want to adopt a pet, or why you do not want to adopt one.",
    "3. Do you own your home or rent?",
    "If renting, do you have permission from your landlord to have pets? If your answer is own, please indicate 'None'.",
    "4. Is your area gated?",
    "If your answer is no, how can we ensure the safety of the pet? If your answer is yes, please indicate 'None'.",
    "5. Will your pet primarily stay indoors or outdoors?",
    "If outdoors, please describe the outdoor setup, including the size of the area, type of shelter provided, and how you will ensure the pet's safety and comfort outdoors. If your answer is indoors, please indicate 'None'.",
    "6. Do you have children in your household?",
    "If yes, please list their ages and how they interact with pets. If your answer is no, please indicate 'None'.",
    "7. Do any household members have allergies to pets?",
    "If yes, how do you manage these allergies? If your answer is no, please indicate 'None'.",
    "8. Is your family prepared to welcome a new family member and commit to their lifelong care?",
    "If yes, please explain how you will take care of the pet. If your answer is no, please explain how you will still take care of the pet.",
    "1. What species of pet have you adopted before? If you have no previous pet adoption experience, please choose 'None'.",
    "Please describe your experience with this pet. If you have no previous pet adoption experience, please indicate 'None'.",
    "2. What was the age category of your pet at the time of adoption? If you have no previous pet adoption experience, please choose 'None'.",
    "Why did you choose this age when adopting your previous pet(s)? If you have no previous pet adoption experience, please indicate 'None'.",
    "3. What was the gender of the pet you have adopted? If you have no previous pet adoption experience, please choose 'None'.",
    "Did the gender influence your choice? Why or why not? If you have no previous pet adoption experience, please indicate 'None'.",
    "4. What is the story behind your adopted pet? If you have no previous pet adoption experience, please choose 'None'.",
    "Please share more details about the adoption/rescue story. If you have no previous pet adoption experience, please indicate 'None'.",
    "5. What was the spay/neuter status of your pet at the time of adoption? If you have no previous pet adoption experience, please choose 'None'.",
    "Why was this important to you? If you have no previous pet adoption experience, please indicate 'None'.",
    "6. What was the vaccination status of your pet at the time of adoption? If you have no previous pet adoption experience, please choose 'None'.",
    "How did you handle their vaccination needs? If you have no previous pet adoption experience, please indicate 'None'.",
    "7. What was the temperament of your adopted pet? If you have no previous pet adoption experience, please choose 'None'.",
    "How did you manage their temperament? If you have no previous pet adoption experience, please indicate 'None'.",
    "1. Vaccinations are highly encouraged for the survival of cats and dogs. Are you willing to provide all needed vaccinations for your new pet?",
    "If yes, please describe your plan for ensuring your new pet receives all necessary vaccinations. If no, please explain why.",
    "2. Are you willing to sponsor the spay and neuter of your adopted pet?",
    "If yes, please describe your plan for arranging and financing the spay and neuter procedure for your new pet. If no, please explain why.",
    "3. If you transfer to a new house/new location, will you bring the pet with you?",
    "If no, please explain under what circumstances you might not bring the pet with you. If your answer is yes, please indicate 'None.'",
    "4. Are you prepared for the long-term commitment of pet ownership, which can extend beyond 10-20 years?",
    "If yes, please explain why you are ready for this commitment and how you plan to provide for the pet's needs over its lifetime. If no, please explain why.",
    "5. We will contact you periodically to get updates on the pet. Is this okay with you?",
    "If no, please specify the reason why. If your answer is yes, please indicate 'None.'",
    "Prediction", "Confidence"
];

const RECOMMENDATION_HEADERS = [
    "Timestamp", "Email address",
    "1. What type of pet are you interested in adopting?",
    "2. What age range of pet are you interested in?",
    "3. Do you have a preference for the pet's gender?",
    "4. What kind of story would you prefer behind your pet's adoption?",
    "5. Would you prefer your pet to be spayed/neutered at the time of adoption?",
    "6. Would you prefer your pet to be vaccinated at the time of adoption?",
    "7. What temperament are you looking for in a pet?"
];

// Utility class to manage temporary files
class TempFileManager {
    constructor() {
        this.files = new Set();
    }

    async createTempDir() {
        await fs.mkdir(TEMP_DIR, { recursive: true });
    }

    track(filePath) {
        this.files.add(filePath);
    }

    async cleanup() {
        for (const file of this.files) {
            try {
                await fs.unlink(file);
                this.files.delete(file);
            } catch (error) {
                console.error(`Failed to delete temp file ${file}:`, error);
            }
        }
    }
}

// Validation functions
const validateAdopterPreferences = (preferences) => {
    const requiredFields = [
        'desired_pet_type',
        'desired_pet_age',
        'desired_pet_gender',
        'desired_pet_story',
        'desired_pet_spay_neuter_status',
        'desired_pet_vaccination_status',
        'desired_pet_temperament'
    ];

    const missingFields = requiredFields.filter(field => !preferences[field]);
    if (missingFields.length > 0) {
        throw new Error(`Missing required fields: ${missingFields.join(', ')}`);
    }
};

// Database interaction functions
async function storeAdopterPreferences(adopterId, preferences) {
    const emailQuery = "SELECT email FROM adopter WHERE adopter_id = ?";
    const [adopter] = await queryAsync(emailQuery, [adopterId]);

    if (!adopter) {
        throw new Error("Adopter not found.");
    }

    const preferenceData = {
        email: adopter.email,
        ...preferences
    };

    const insertQuery = `
        INSERT INTO recommendations (
            email, desired_pet_type, desired_pet_age, desired_pet_gender,
            desired_pet_story, desired_pet_spay_neuter_status,
            desired_pet_vaccination_status, desired_pet_temperament, timestamp
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, NOW())
    `;

    await queryAsync(insertQuery, [
        preferenceData.email,
        preferenceData.desired_pet_type,
        preferenceData.desired_pet_age,
        preferenceData.desired_pet_gender,
        preferenceData.desired_pet_story,
        preferenceData.desired_pet_spay_neuter_status,
        preferenceData.desired_pet_vaccination_status,
        preferenceData.desired_pet_temperament
    ]);

    return preferenceData;
}

async function getScreeningData(adopterId) {
    const screeningQuery = "SELECT * FROM screenings WHERE adopter_id = ?";
    const [screening] = await queryAsync(screeningQuery, [adopterId]);

    if (!screening) {
        throw new Error("Screening data not found.");
    }

    const [adopter] = await queryAsync("SELECT * FROM adopter WHERE adopter_id = ?", [screening.adopter_id]);
    const [personalInfo] = await queryAsync("SELECT * FROM personal_infos WHERE personal_info_id = ?", [screening.personal_info_id]);
    const [livingSituation] = await queryAsync("SELECT * FROM living_situations WHERE living_situation_id = ?", [screening.living_situation_id]);
    const [prevPetAdoption] = await queryAsync("SELECT * FROM prev_pet_adoption_experiences WHERE prev_pet_adoption_experience_id = ?", [screening.prev_pet_adoption_experience_id]);
    const [commitmentToPet] = await queryAsync("SELECT * FROM commitment_to_pet_ownerships WHERE commitment_to_pet_ownership_id = ?", [screening.commitment_to_pet_ownership_id]);

    return {
        screening,
        adopter,
        personalInfo,
        livingSituation,
        prevPetAdoption,
        commitmentToPet
    };
}

async function getPetTypesData() {
    const petTypesQuery = "SELECT * FROM pet_types";
    const petTypes = await queryAsync(petTypesQuery);

    const formattedHeaders = {
        type: "Type",
        age: "Age",
        gender: "Gender",
        vaccinated: "Vaccinated",
        spayed_neutered: "Spayed/Neutered",
        adoption_story: "AdoptionStory",
        extracted_temperaments: "Extracted_Temperaments"
    };

    return petTypes.map(({ pet_type_id, ...rest }) => {
        const formattedRow = {};
        Object.keys(rest).forEach((key) => {
            if (formattedHeaders[key]) {
                formattedRow[formattedHeaders[key]] = rest[key];
            }
        });
        return formattedRow;
    });
}

// File handling functions
async function createTemporaryCsv(data, headers, fileName, tempFileManager) {
    if (!data?.length) {
        throw new Error(`No data provided for ${fileName}`);
    }

    const tempFilePath = path.join(TEMP_DIR, `${fileName}_${Date.now()}.csv`);
    tempFileManager.track(tempFilePath);

    const csvData = await new Promise((resolve, reject) => {
        csv.stringify(data, { header: true, columns: headers }, (err, output) => {
            if (err) reject(err);
            else resolve(output);
        });
    });

    await fs.writeFile(tempFilePath, csvData);
    return tempFilePath;
}

// Python script execution
function executePythonScript(scriptPath, args) {
    return new Promise((resolve, reject) => {
        const python = spawn("python3", [scriptPath, ...args]);
        let output = "";
        let errorOutput = "";

        const timeout = setTimeout(() => {
            python.kill();
            reject(new Error("Python script execution timed out"));
        }, PYTHON_TIMEOUT);

        python.stdout.on("data", (data) => {
            output += data.toString();
        });

        python.stderr.on("data", (data) => {
            errorOutput += data.toString();
            console.error("Python Error:", data.toString());
        });

        python.on("close", (code) => {
            clearTimeout(timeout);
            if (code === 0) {
                resolve(output);
            } else {
                reject(new Error(`Python script failed with code ${code}: ${errorOutput}`));
            }
        });

        python.on("error", (error) => {
            clearTimeout(timeout);
            reject(error);
        });
    });
}

// Data mapping functions
function mapScreeningDataToRow(screeningData) {
    const { screening, adopter, personalInfo, livingSituation, prevPetAdoption, commitmentToPet } = screeningData;

    const now = new Date();
    const timestamp = `${now.getFullYear()}/${
        String(now.getMonth() + 1).padStart(2, "0")
    }/${String(now.getDate()).padStart(2, "0")} ${
        now.getHours() % 12 || 12
    }:${String(now.getMinutes()).padStart(2, "0")}:${String(
        now.getSeconds()
    ).padStart(2, "0")} ${
        now.getHours() >= 12 ? "pm" : "am"
    } GMT+8`;

    const cleanValue = (value) => (value === "None" || value === null || value === undefined ? "" : value);

    return {
        Timestamp: timestamp,
        Username: cleanValue(adopter.email),
        "Full Name:": cleanValue(personalInfo.fullname),
        "Age:": cleanValue(personalInfo.age),
        "Gender:": cleanValue(personalInfo.gender),
        "Civil Status:": cleanValue(personalInfo.civil_status),
        // Living situation fields
        "1. Are you employed?": cleanValue(livingSituation.employment_status),
        "If yes, please describe your employment situation (e.g., full-time, part-time, remote, etc.) and how it supports your ability to care for a pet. If your answer is no, please indicate 'None'.": cleanValue(livingSituation.employment_status_reason),
        "2. Are you genuinely interested in adopting a pet?": cleanValue(livingSituation.interest_in_adoption),
        "Please explain why you want to adopt a pet, or why you do not want to adopt one.": cleanValue(livingSituation.interest_in_adoption_reason),
        "3. Do you own your home or rent?": cleanValue(livingSituation.housing_ownership),
        "If renting, do you have permission from your landlord to have pets? If your answer is own, please indicate 'None'.": cleanValue(livingSituation.housing_ownership_reason),
        "4. Is your area gated?": cleanValue(livingSituation.gated_community_status),
        "If your answer is no, how can we ensure the safety of the pet? If your answer is yes, please indicate 'None'.": cleanValue(livingSituation.gated_community_status_reason),
        "5. Will your pet primarily stay indoors or outdoors?": cleanValue(livingSituation.pet_living_environment),
        "If outdoors, please describe the outdoor setup, including the size of the area, type of shelter provided, and how you will ensure the pet's safety and comfort outdoors. If your answer is indoors, please indicate 'None'.": cleanValue(livingSituation.pet_living_environment_reason),
        "6. Do you have children in your household?": cleanValue(livingSituation.children_in_household),
        "If yes, please list their ages and how they interact with pets. If your answer is no, please indicate 'None'.": cleanValue(livingSituation.children_in_household_reason),
        "7. Do any household members have allergies to pets?": cleanValue(livingSituation.household_pet_allergies),
        "If yes, how do you manage these allergies? If your answer is no, please indicate 'None'.": cleanValue(livingSituation.household_pet_allergies_reason),
        "8. Is your family prepared to welcome a new family member and commit to their lifelong care?": cleanValue(livingSituation.family_commitment_to_pet),
        "If yes, please explain how you will take care of the pet. If your answer is no, please explain how you will still take care of the pet.": cleanValue(livingSituation.family_commitment_to_pet_reason),
        // Previous pet adoption experience fields
        "1. What species of pet have you adopted before? If you have no previous pet adoption experience, please choose 'None'.": cleanValue(prevPetAdoption.previous_pet_species),
        "Please describe your experience with this pet. If you have no previous pet adoption experience, please indicate 'None'.": cleanValue(prevPetAdoption.previous_pet_species_details),
        "2. What was the age category of your pet at the time of adoption? If you have no previous pet adoption experience, please choose 'None'.": cleanValue(prevPetAdoption.previous_pet_age_category),
        "Why did you choose this age when adopting your previous pet(s)? If you have no previous pet adoption experience, please indicate 'None'.": cleanValue(prevPetAdoption.previous_pet_age_category_details),
        "3. What was the gender of the pet you have adopted? If you have no previous pet adoption experience, please choose 'None'.": cleanValue(prevPetAdoption.previous_pet_gender),
        "Did the gender influence your choice? Why or why not? If you have no previous pet adoption experience, please indicate 'None'.": cleanValue(prevPetAdoption.previous_pet_gender_details),
        "4. What is the story behind your adopted pet? If you have no previous pet adoption experience, please choose 'None'.": cleanValue(prevPetAdoption.previous_pet_history),
        "Please share more details about the adoption/rescue story. If you have no previous pet adoption experience, please indicate 'None'.": cleanValue(prevPetAdoption.previous_pet_history_details),
        "5. What was the spay/neuter status of your pet at the time of adoption? If you have no previous pet adoption experience, please choose 'None'.": cleanValue(prevPetAdoption.previous_pet_spay_neuter_status),
        "Why was this important to you? If you have no previous pet adoption experience, please indicate 'None'.": cleanValue(prevPetAdoption.previous_pet_spay_neuter_status_details),
        "6. What was the vaccination status of your pet at the time of adoption? If you have no previous pet adoption experience, please choose 'None'.": cleanValue(prevPetAdoption.previous_pet_vaccinations_status),
        "How did you handle their vaccination needs? If you have no previous pet adoption experience, please indicate 'None'.": cleanValue(prevPetAdoption.previous_pet_vaccinations_status_details),
        "7. What was the temperament of your adopted pet? If you have no previous pet adoption experience, please choose 'None'.": cleanValue(prevPetAdoption.previous_pet_temperament),
        "How did you manage their temperament? If you have no previous pet adoption experience, please indicate 'None'.": cleanValue(prevPetAdoption.previous_pet_temperament_details),
        // Commitment to pet ownership fields
        "1. Vaccinations are highly encouraged for the survival of cats and dogs. Are you willing to provide all needed vaccinations for your new pet?": cleanValue(commitmentToPet.new_pet_vaccination_willingness),
        "If yes, please describe your plan for ensuring your new pet receives all necessary vaccinations. If no, please explain why.": cleanValue(commitmentToPet.new_pet_vaccination_willingness_description),
        "2. Are you willing to sponsor the spay and neuter of your adopted pet?": cleanValue(commitmentToPet.spay_neuter_willingness),
        "If yes, please describe your plan for arranging and financing the spay and neuter procedure for your new pet. If no, please explain why.": cleanValue(commitmentToPet.spay_neuter_willingness_description),
        "3. If you transfer to a new house/new location, will you bring the pet with you?": cleanValue(commitmentToPet.pet_transfer_willingness),
        "If no, please explain under what circumstances you might not bring the pet with you. If your answer is yes, please indicate 'None.'": cleanValue(commitmentToPet.pet_transfer_willingness_description),
        "4. Are you prepared for the long-term commitment of pet ownership, which can extend beyond 10-20 years?": cleanValue(commitmentToPet.new_pet_long_term_commitment),
        "If yes, please explain why you are ready for this commitment and how you plan to provide for the pet's needs over its lifetime. If no, please explain why.": cleanValue(commitmentToPet.new_pet_long_term_commitment_description),
        "5. We will contact you periodically to get updates on the pet. Is this okay with you?": cleanValue(commitmentToPet.follow_up_permission),
        "If no, please specify the reason why. If your answer is yes, please indicate 'None.'": cleanValue(commitmentToPet.follow_up_permission_description),
        // Prediction and confidence
        Prediction: cleanValue(screening.status),
        Confidence: cleanValue(screening.confidence)
    };
}

function mapPreferenceData(preferenceData) {
    const now = new Date();
    const timestamp = `${String(now.getDate()).padStart(2, "0")}/${
        String(now.getMonth() + 1).padStart(2, "0")
    }/${now.getFullYear()} ${String(now.getHours()).padStart(2, "0")}:${String(now.getMinutes()).padStart(2, "0")}`;
    return {
        Timestamp: timestamp,
        "Email address": preferenceData.email,
        "1. What type of pet are you interested in adopting?": preferenceData.desired_pet_type,
        "2. What age range of pet are you interested in?": preferenceData.desired_pet_age,
        "3. Do you have a preference for the pet's gender?": preferenceData.desired_pet_gender,
        "4. What kind of story would you prefer behind your pet's adoption?": preferenceData.desired_pet_story,
        "5. Would you prefer your pet to be spayed/neutered at the time of adoption?": preferenceData.desired_pet_spay_neuter_status,
        "6. Would you prefer your pet to be vaccinated at the time of adoption?": preferenceData.desired_pet_vaccination_status,
        "7. What temperament are you looking for in a pet?": preferenceData.desired_pet_temperament
    };
}

// Main controller function
exports.processAndRecommend = async (req, res) => {
    const tempFileManager = new TempFileManager();

    try {
        const adopterId = parseInt(req.params.adopterId);
        if (isNaN(adopterId) || adopterId <= 0) {
            throw new Error("Invalid adopter ID");
        }

        const preferences = await storeAdopterPreferences(adopterId, req.body);
        const screeningData = await getScreeningData(adopterId);
        const petTypesData = await getPetTypesData();

        const [preferencesFile, screeningFile, petTypesFile] = await Promise.all([
            createTemporaryCsv([mapPreferenceData(preferences)], RECOMMENDATION_HEADERS, "preferences", tempFileManager),
            createTemporaryCsv([mapScreeningDataToRow(screeningData)], SCREENING_HEADERS, "screening", tempFileManager),
            createTemporaryCsv(petTypesData, Object.keys(petTypesData[0]), "pets", tempFileManager)
        ]);

        const pythonScript = path.join(__dirname, "../recommending_model/reco.py");
        const output = await executePythonScript(pythonScript, [
            preferencesFile,
            screeningFile,
            petTypesFile,
            adopterId.toString(), // Pass adopterId as a string
        ]);
        

        const recommendations = JSON.parse(output);

        if (recommendations.status === "error") {
            throw new Error(recommendations.message);
        }

        res.status(200).json({
            success: true,
            message: "Recommendations generated successfully",
            data: recommendations.recommendations
        });

    } catch (error) {
        console.error("Error generating recommendations:", error);
        res.status(500).json({ error: error.message || "Failed to generate recommendations" });
    } finally {
        await tempFileManager.cleanup();
    }
};