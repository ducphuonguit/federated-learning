"use client"
import { useMemo, useState } from 'react';
import axios from 'axios';
import { Button } from './button';
import { Input } from './input';
import { Card } from './card';
import { toast } from "sonner";
import { AnimatePresence, motion } from "framer-motion";
const ImageUpload = () => {
    const [selectedFile, setSelectedFile] = useState<File | null>(null);
    const [prediction, setPrediction] = useState<string | null>(null);
    const [error, setError] = useState<any>(null);
    const selectedImage = useMemo(() => {
        if (selectedFile) {
            return URL.createObjectURL(selectedFile)
        }
        return null
    }, [selectedFile])
    const handleFileChange = (event: any) => {
        setSelectedFile(event.target.files[0]);
        setPrediction(null); // Reset prediction on new file selection
        setError(null);      // Reset error
    };

    const handleSubmit = async (event: any) => {
        event.preventDefault();
        if (!selectedFile) {
            setError("Please select an image file.");
            return;
        }

        const formData = new FormData();
        formData.append('image', selectedFile);

        try {
            const response = await axios.post('http://127.0.0.1:8081/predict', formData, {
                headers: {
                    'Content-Type': 'multipart/form-data',
                },
            });
            setPrediction(response.data.predicted);
            toast.success("Predicted successfully")
        } catch (error) {
            console.error("Error uploading file:", error);
            setError("Failed to predict the image. Please try again.");
        }
    };

    return (
        <Card className='rounded-lg shadow-xl min-h-[510px]'>
            <div className='flex flex-col items-center justify-center space-y-4 p-10 gap-5'>
                <h1 className="scroll-m-20 text-4xl font-extrabold tracking-tight lg:text-5xl">IMAGE PREDICTION</h1>
                <form onSubmit={handleSubmit} className='flex gap-4'>
                    <Input id="fileInput" type="file" accept="image/*" onChange={handleFileChange} className='cursor-pointer bg-white' />
                    <Button type="submit">Predict</Button>
                </form>
                <AnimatePresence>
                    {!!selectedImage && (
                        <motion.div
                            key={selectedImage}
                            // exit={{ opacity: 0, scale: 1.1 }}
                            initial={{ opacity: 0, scale: 0.95 }}
                            className="flex gap-2"
                            animate={{ opacity: 1, scale: 1 }}
                        >
                            <img src={selectedImage} alt="Selected" style={{ width: '200px', height: 'auto' }} />
                        </motion.div>
                    )}
                </AnimatePresence>
                {prediction !== null && <p>Predicted Value: {prediction}</p>}
                {error && <span style={{ color: 'red' }}>{error.toString()}</span>}
            </div>
        </Card>

    );
};

export default ImageUpload;
