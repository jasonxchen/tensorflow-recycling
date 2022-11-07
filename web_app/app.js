import express from "express";

const app = express();
const PORT = 8000;
app.set("view engine", "ejs");

app.use(express.static("./"));

// Home page
app.get("/", (req, res) => {
    res.render("index.ejs");
});

// Route for serving the model
app.get("/model", (req, res) => {
    const options = {
        root: "./"
    };
    res.sendFile("./model.json", options);
});

app.listen(PORT, () => {
    console.log(`Conected on port ${PORT}`);
});
