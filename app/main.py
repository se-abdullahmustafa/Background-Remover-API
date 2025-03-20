# Import required libraries
from fastapi import FastAPI, File, UploadFile, BackgroundTasks
from fastapi.responses import FileResponse
from PIL import Image
import torch
from torchvision import transforms
from transformers import AutoModelForImageSegmentation
import os
import time
from datetime import datetime
import logging
from pathlib import Path
import asyncio
import glob
from contextlib import asynccontextmanager
import uvicorn

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Create upload and output directories
upload_dir = Path("uploads")
output_dir = Path("outputs")
upload_dir.mkdir(exist_ok=True)
output_dir.mkdir(exist_ok=True)

# Global variable to store the cleanup task
cleanup_task = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: load model and start background task
    global birefnet, device, cleanup_task

    # Load the BiRefNet model
    logger.info("Loading model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    birefnet = AutoModelForImageSegmentation.from_pretrained(
        "zhengpeng7/BiRefNet", trust_remote_code=True
    )
    birefnet.to(device)
    birefnet.eval()
    logger.info(f"Model loaded successfully using {device}")

    # Start the cleanup task
    logger.info("Starting background task for file cleanup")
    cleanup_task = asyncio.create_task(delete_old_files())

    yield  # FastAPI runs the application here

    # Shutdown: cancel the cleanup task
    if cleanup_task:
        logger.info("Cancelling cleanup task")
        cleanup_task.cancel()
        try:
            await cleanup_task
        except asyncio.CancelledError:
            logger.info("Cleanup task cancelled successfully")


# Initialize FastAPI app with lifespan
app = FastAPI(lifespan=lifespan)

# Image transformation
transform_image = transforms.Compose(
    [
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def refine_foreground(image, mask):
    """
    Refine the foreground of the image using the mask.
    """
    # Convert mask to RGBA
    mask = mask.convert("L")
    mask = mask.point(lambda p: p > 128 and 255)  # Binarize the mask
    mask = mask.convert("RGBA")

    # Apply the mask to the image
    image_rgba = image.convert("RGBA")
    image_masked = Image.composite(
        image_rgba, Image.new("RGBA", image.size, (0, 0, 0, 0)), mask
    )
    return image_masked


async def delete_old_files():
    """
    Background task to delete files older than 1 hour
    """
    while True:
        try:
            # Sleep for 10 minutes before checking files
            await asyncio.sleep(600)  # 10 minutes in seconds

            current_time = time.time()
            one_hour_in_seconds = 3600

            # Delete old files from upload directory
            for file_path in glob.glob(str(upload_dir / "*")):
                if os.path.isfile(file_path):
                    file_age = current_time - os.path.getmtime(file_path)
                    if file_age > one_hour_in_seconds:
                        try:
                            os.remove(file_path)
                            logger.info(f"Deleted old upload file: {file_path}")
                        except Exception as e:
                            logger.error(f"Error deleting file {file_path}: {str(e)}")

            # Delete old files from output directory
            for file_path in glob.glob(str(output_dir / "*")):
                if os.path.isfile(file_path):
                    file_age = current_time - os.path.getmtime(file_path)
                    if file_age > one_hour_in_seconds:
                        try:
                            os.remove(file_path)
                            logger.info(f"Deleted old output file: {file_path}")
                        except Exception as e:
                            logger.error(f"Error deleting file {file_path}: {str(e)}")

        except asyncio.CancelledError:
            logger.info("File cleanup task was cancelled")
            break
        except Exception as e:
            logger.error(f"Error in delete_old_files task: {str(e)}")
            # Sleep a bit to avoid tight error loops
            await asyncio.sleep(60)


@app.post("/remove-bg/")
async def process_image(file: UploadFile = File(...)):
    """
    API endpoint to process an image using BiRefNet.
    Returns only the refined foreground image with timestamp.
    """
    try:
        # Generate timestamp for file names
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create file paths with timestamp
        input_path = upload_dir / f"input_{timestamp}.jpg"
        output_path = output_dir / f"subject_{timestamp}.png"

        # Save the uploaded file with timestamp
        with open(input_path, "wb") as buffer:
            buffer.write(await file.read())

        # Load and process the image
        image = Image.open(input_path)
        input_images = transform_image(image).unsqueeze(0).to(device)

        # Prediction
        with torch.no_grad():
            preds = birefnet(input_images)[-1].sigmoid().cpu()
        pred = preds[0].squeeze()

        # Save the results
        pred_pil = transforms.ToPILImage()(pred)
        pred_pil = pred_pil.resize(image.size)

        # Refine the foreground and save the subject image
        image_masked = refine_foreground(image, pred_pil)
        image_masked.putalpha(pred_pil)
        image_masked.save(output_path)

        logger.info(f"Processed image saved as {output_path}")

        # Return the subject image
        return FileResponse(
            output_path, media_type="image/png", filename=f"subject_{timestamp}.png"
        )

    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        return {"error": str(e)}


# Health check endpoint
@app.get("/health")
async def health_check():
    """Simple health check endpoint"""
    return {"status": "healthy", "device": device}


# Run the FastAPI app
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
