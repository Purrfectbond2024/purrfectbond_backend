const express = require("express");
const form = require("../controllers/createFormController");
const recommendationController = require("../controllers/createRecommendationController");

const router = express.Router();

// Middleware for async error handling
const asyncHandler = fn => (req, res, next) => {
  Promise.resolve(fn(req, res, next)).catch(next);
};

// Middleware for request validation
const validateAdopterId = (req, res, next) => {
  const adopterId = parseInt(req.params.adopterId);
  if (isNaN(adopterId) || adopterId <= 0) {
    return res.status(400).json({
      success: false,
      error: 'Invalid adopter ID provided'
    });
  }
  req.params.adopterId = adopterId;
  next();
};

// Form submission routes
router.post(
  '/request/form',
  asyncHandler(form.createForm)
);

// Pet recommendation routes
router.post(
  '/recommendations/:adopterId',
  validateAdopterId,
  asyncHandler(recommendationController.processAndRecommend)
);

// Error handling middleware
router.use((err, req, res, next) => {
  console.error('Route Error:', err);
  res.status(500).json({
    success: false,
    error: err.message || 'An unexpected error occurred'
  });
});

module.exports = router;

