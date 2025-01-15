"use client";
import { useState, useEffect } from "react";
import axios from "axios";
import { Button } from "./button";
import { Input } from "./input";
import { Card } from "./card";
import { toast } from "sonner";

const MnistUpload = () => {
    const [files, setFiles] = useState<{ [key: string]: File | null }>({
        trainImages: null,
        trainLabels: null,
        testImages: null,
        testLabels: null,
    });

    const [status, setStatus] = useState<string | null>(null);
    const [polling, setPolling] = useState<boolean>(false);

    const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>, type: string) => {
        const file = event.target.files?.[0] || null;
        setFiles((prevFiles) => ({ ...prevFiles, [type]: file }));
    };

    const handleSubmit = async (event: React.FormEvent) => {
        event.preventDefault();

        // Check if all files are uploaded
        if (!files.trainImages || !files.trainLabels || !files.testImages || !files.testLabels) {
            toast.error("All four files are required.");
            return;
        }

        const formData = new FormData();
        formData.append("trainImages", files.trainImages, "train-images-idx3-ubyte.gz");
        formData.append("trainLabels", files.trainLabels, "train-labels-idx1-ubyte.gz");
        formData.append("testImages", files.testImages, "t10k-images-idx3-ubyte.gz");
        formData.append("testLabels", files.testLabels, "t10k-labels-idx1-ubyte.gz");

        try {
            const response = await axios.post("http://127.0.0.1:4001/train", formData, {
                headers: {
                    "Content-Type": "multipart/form-data",
                },
            });
            setStatus(response.data.status);
            setPolling(true); // Start polling after submission
            toast.success("Files uploaded successfully!");
        } catch (error) {
            console.error("Error uploading files:", error);
            toast.error("Failed to upload files.");
        }
    };

    useEffect(() => {
        let interval: NodeJS.Timeout | null = null;

        if (polling) {
            interval = setInterval(async () => {
                try {
                    const response = await axios.get("http://127.0.0.1:4000/status");
                    setStatus(response.data.status);

                    if (response.data.status === "idle") {
                        setPolling(false); // Stop polling
                        toast.success("Training completed!");
                    }
                } catch (error) {
                    console.error("Error fetching status:", error);
                    toast.error("Failed to fetch status.");
                }
            }, 1000);
        }

        return () => {
            if (interval) {
                clearInterval(interval);
            }
        };
    }, [polling]);

    return (
        <Card className="rounded-lg shadow-xl min-h-[510px]">
            <div className="flex flex-col items-center justify-center space-y-4 p-10 gap-5">
                <div className="flex items-center justify-center w-full flex-col">
                    <h1 className="scroll-m-20 text-4xl font-extrabold tracking-tight lg:text-5xl text-center">
                        Upload MNIST Data

                    </h1>
                    <h1 className="scroll-m-20 text-4xl font-extrabold tracking-tight lg:text-5xl text-center">
                        Client 2
                    </h1>
                </div>

                <form onSubmit={handleSubmit} className="flex flex-col gap-4">
                    <Input
                        type="file"
                        onChange={(e) => handleFileChange(e, "trainImages")}
                    />
                    <Input
                        type="file"
                        onChange={(e) => handleFileChange(e, "trainLabels")}
                    />
                    <Input
                        type="file"
                        onChange={(e) => handleFileChange(e, "testImages")}
                    />
                    <Input
                        type="file"
                        onChange={(e) => handleFileChange(e, "testLabels")}
                    />
                    <Button type="submit">Upload Files</Button>
                </form>
                {status && <p>Status: {status}</p>}
            </div>
        </Card>
    );
};

export default MnistUpload;
