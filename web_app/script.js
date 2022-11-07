// import * as tf from "@tensorflow/tfjs"

// DOM selectors
const STATUS = document.getElementById("status");
const MAIN = document.querySelector("main");
const IMAGEUP = document.getElementById("imageUp");
const PREDICT_BUTTON = document.getElementById("predict");
const IMAGE = document.getElementById("image");
const PREVIEW = document.getElementById("preview");
const INFO = document.getElementById("info");
const NAME = document.getElementById("name");
const USES = document.getElementById("uses");
const RECYCLE = document.getElementById("recycle");

// Global variables
const INPUT_WIDTH = 192;
const INPUT_HEIGHT = 192;
const CLASS_NAMES = ["1 - PET", "2 - HDPE", "3 - PVC", "4 - LDPE", "5 - PP", "6 - PS", "7 - OTHER"];
let model = undefined;
let userImage = undefined;
let predict = false;

// Event listeners
PREDICT_BUTTON.addEventListener("click", makePrediction);

IMAGEUP.addEventListener("change", e => {
	IMAGE.src = URL.createObjectURL(e.target.files[0]);
	PREVIEW.src = URL.createObjectURL(e.target.files[0]);
	IMAGE.onload = () => {
		URL.revokeObjectURL(IMAGE.src);
	}
	PREVIEW.onload = () => {
		URL.revokeObjectURL(PREVIEW.src);
	}
})

// Have the model make run inference on the uploaded image
function makePrediction() {
	if (predict) {
		// tidy function cleans up memory after inference is completed
		tf.tidy(function() {
			// converts the HTML image element into a tensor
			userImage = tf.browser.fromPixels(IMAGE).div(1.0);
			// resize the tensor into useable input
			let resizedTensor = tf.image.resizeBilinear(userImage, [INPUT_WIDTH, INPUT_HEIGHT], true);
			// run the prediction
			let prediction = model.predict(resizedTensor.expandDims()).softmax().squeeze();
			// console.log(prediction.print())
			// grab the highest percent prediction
			let highestIndex = prediction.argMax().arraySync();
			let predictionArray = prediction.arraySync();
			
			// set status message to the predicted class and percent confidence
			STATUS.innerText = "Prediction: " + CLASS_NAMES[highestIndex] + " with " + Math.floor(predictionArray[highestIndex] * 100) + "% confidence";

			// switch case to handle display of info on the resin code
			switch(highestIndex) {
				case (0):
					NAME.innerText = "Polyethylene terephthalate";
					USES.innerText = "Polyester fibres (Polar Fleece), thermoformed sheet, strapping, soft drink bottles, tote bags, furniture, carpet, paneling and (occasionally) new containers.";
					RECYCLE.innerText = "Picked up through most curbside recycling programs.";
					break;
				case (1):
					NAME.innerText = "High-density polyethylene";
					USES.innerText = "Bottles, grocery bags, milk jugs, recycling bins, agricultural pipe, base cups, car stops, playground equipment, and plastic lumber";
					RECYCLE.innerText = "Picked up through most curbside recycling programs, although some allow only those containers with necks.";
					break;
				case (2):
					NAME.innerText = "Polyvinyl chloride";
					USES.innerText = "Pipe, window profile, siding, fencing, flooring, shower curtains, lawn chairs, non-food bottles, and children's toys.";
					RECYCLE.innerText = "Too long life for significant recycling volumes although there was 740,000 tonnes recycled in 2018 through EU Vinyl 2010 and VinylPlus initiatives.";
					break;
				case (3):
					NAME.innerText = "Low-density polyethylene";
					USES.innerText = "Plastic bags, six pack rings, various containers, dispensing bottles, wash bottles, tubing, and various molded laboratory equipment";
					RECYCLE.innerText = "LDPE is not often recycled through curbside programs and is a significant source of plastic pollution. LDPE can often be returned to many stores for recycling.";
					break;
				case (4):
					NAME.innerText = "Polypropylene";
					USES.innerText = "Auto parts, industrial fibres, food containers, and dishware";
					RECYCLE.innerText = "Picked up through most curbside recycling programs.";
					break;
				case (5):
					NAME.innerText = "Polystyrene";
					USES.innerText = "Desk accessories, cafeteria trays, plastic utensils, coffee cup lids, toys, video cassettes and cases, clamshell containers, packaging peanuts, and insulation board and other expanded polystyrene products (e.g., Styrofoam)";
					RECYCLE.innerText = "Polystyrene is often not recycled through curbside programs as it is too lightweight to be economical to recycle, usually incinerated instead.";
					break;
				case (6):
					NAME.innerText = "Other plastics, such as acrylic, nylon, polycarbonate, and polylactic acid (a bioplastic also known as PLA), and multilayer combinations of different plastics";
					USES.innerText = "Bottles, plastic lumber applications, headlight lenses, and safety shields/glasses.";
					RECYCLE.innerText = "Number 7 plastics are not typically recycled as they were mostly specialty produced in limited volumes at the time the codes were established.";
					break;
				default:
					break;
			}

			// remove hidden attribute from info div
			INFO.hidden = false;
		});
	}
}

/**
 * Loads the model and warms it up so ready for use.
 **/
async function loadModel() {
	const URL = "http://localhost:8000/model";

	model = await tf.loadGraphModel(URL);
	
	// Warm up the model by passing zeros through it once.
	tf.tidy(function () {
		let answer = model.predict(tf.zeros([1, INPUT_WIDTH, INPUT_HEIGHT, 3]));
		console.log("Warm up test", answer.softmax().squeeze().print());
	});

	// Inform user model is ready and show the rest of page
	STATUS.innerText = "Model loaded successfully!";
	predict = true;
	MAIN.hidden = false;
}

// Call the function immediately to start loading.
loadModel();
