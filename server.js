const express = require("express");
const cors = require("cors");
const formRoutes = require("./routes/form")
require("./db/index");

const app = express();

var corsOptions = {
    origin: ["http://localhost:8081", "http://localhost:3000"], // Allow both React dev server and your frontend
    methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
    allowedHeaders: ['Content-Type', 'Accept']
};

app.use(cors(corsOptions));
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

app.use(formRoutes);

app.get("/", (req, res) => {
    res.json({ message: "Welcome to the Backend!" });
});

const PORT = process.env.PORT || 8080;

app.listen(PORT, () => {
    console.log(`Server is running on port ${PORT}`);
});