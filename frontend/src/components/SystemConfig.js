import { useState, useEffect } from "react";
import axios from "axios";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Server, Cpu, ShieldCheck, AlertCircle, CheckCircle2, Loader2, Activity } from "lucide-react";
import { toast } from "sonner";

const SystemConfig = ({ apiUrl }) => {
    const [catalog, setCatalog] = useState(null);
    const [currentModel, setCurrentModel] = useState("unknown");

    const [selectedTier, setSelectedTier] = useState("mid");
    const [selectedModel, setSelectedModel] = useState("");

    const [switching, setSwitching] = useState(false);
    const [jobId, setJobId] = useState(null);
    const [jobStatus, setJobStatus] = useState(null); // { status, details }

    const [loading, setLoading] = useState(true);

    // Hardcoded system limit for visualization context (e.g. 8GB standard laptop)
    // In a real app, this could come from a /health/system-info endpoint
    const SYSTEM_RAM_GB = 8;

    const fetchCatalog = async () => {
        try {
            setLoading(true);
            const res = await axios.get(`${apiUrl}/models`);
            setCatalog(res.data.catalog);
            setCurrentModel(res.data.current);

            // Auto-select current tier if possible
            let found = false;
            if (res.data.catalog && res.data.current) {
                for (const [tier, models] of Object.entries(res.data.catalog)) {
                    const match = models.find(m => m.name === res.data.current);
                    if (match) {
                        setSelectedTier(tier);
                        setSelectedModel(match.name);
                        found = true;
                        break;
                    }
                }
            }

            if (!found && res.data.catalog?.mid) {
                setSelectedTier("mid");
                setSelectedModel(res.data.catalog.mid[0].name);
            }

        } catch (err) {
            console.error("Failed to fetch catalog", err);
            toast.error("Failed to load model catalog");
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        fetchCatalog();
    }, [apiUrl]);

    // Polling for Job Status
    useEffect(() => {
        let interval;
        if (jobId && switching) {
            interval = setInterval(async () => {
                try {
                    const res = await axios.get(`${apiUrl}/model/status/${jobId}`);
                    const status = res.data.status;
                    setJobStatus(res.data);

                    if (status === "success") {
                        toast.success(res.data.details || "Model switched successfully");
                        setSwitching(false);
                        setJobId(null);
                        fetchCatalog(); // Refresh current model
                    } else if (status === "failed") {
                        toast.error(res.data.details || "Model switch failed");
                        setSwitching(false);
                        setJobId(null);
                    }
                    // if running, keep polling
                } catch (err) {
                    console.error("Poll failed", err);
                    // Don't stop polling immediately on one error, network might be flaky
                }
            }, 2000); // 2s poll
        }
        return () => clearInterval(interval);
    }, [jobId, switching, apiUrl]);

    const handleSetModel = async () => {
        if (!selectedModel) return;

        try {
            setSwitching(true);
            setJobStatus({ status: "starting", details: "Initiating switch..." });
            const res = await axios.post(`${apiUrl}/model/set`, { model: selectedModel });
            setJobId(res.data.job_id);
        } catch (err) {
            console.error(err);
            toast.error(err.response?.data?.detail || "Failed to start model switch");
            setSwitching(false);
        }
    };

    const getRamColor = (gb) => {
        const ratio = gb / SYSTEM_RAM_GB;
        if (ratio > 1.2) return "bg-red-500"; // Severe warning
        if (ratio > 0.9) return "bg-amber-500"; // Warning
        return "bg-green-500"; // OK
    };

    const getRamText = (gb) => {
        const ratio = gb / SYSTEM_RAM_GB;
        if (ratio > 1.2) return "Requires High-Spec (Might Freeze)";
        if (ratio > 0.9) return "High Usage (Close other apps)";
        return "Optimal";
    };

    // Safe accessor for current tier's models
    const tierModels = catalog ? (catalog[selectedTier] || []) : [];
    const activeModelObj = tierModels.find(m => m.name === selectedModel);

    if (loading) return (
        <Card className="mb-6 animate-pulse">
            <CardHeader><div className="h-6 bg-slate-200 rounded w-1/3"></div></CardHeader>
            <CardContent><div className="h-20 bg-slate-100 rounded"></div></CardContent>
        </Card>
    );

    return (
        <Card className="mb-6 border-slate-200 shadow-sm bg-white/80 backdrop-blur">
            <CardHeader className="pb-3">
                <div className="flex items-center justify-between">
                    <CardTitle className="flex items-center gap-2 text-lg">
                        <Server className="w-5 h-5 text-indigo-600" />
                        System Configuration
                    </CardTitle>
                    <Badge variant="outline" className="flex items-center gap-1 font-mono">
                        <Activity className="w-3 h-3" />
                        Active: {currentModel}
                    </Badge>
                </div>
                <CardDescription>
                    Select your AI Model Tier based on available hardware
                </CardDescription>
            </CardHeader>

            <CardContent className="space-y-4">
                {/* Selection Row */}
                <div className="grid grid-cols-1 md:grid-cols-12 gap-4 items-end">
                    {/* Tier Selector */}
                    <div className="md:col-span-3 space-y-2">
                        <label className="text-sm font-medium text-slate-700">Performance Tier</label>
                        <Select value={selectedTier} onValueChange={(val) => { setSelectedTier(val); setSelectedModel(""); }} disabled={switching}>
                            <SelectTrigger>
                                <SelectValue placeholder="Select Tier" />
                            </SelectTrigger>
                            <SelectContent>
                                <SelectItem value="low">Low (1-3GB RAM)</SelectItem>
                                <SelectItem value="mid">Mid (4-8GB RAM)</SelectItem>
                                <SelectItem value="high">High (16GB+ RAM)</SelectItem>
                            </SelectContent>
                        </Select>
                    </div>

                    {/* Model Selector */}
                    <div className="md:col-span-5 space-y-2">
                        <label className="text-sm font-medium text-slate-700">Model Selection</label>
                        <Select value={selectedModel} onValueChange={setSelectedModel} disabled={switching || !catalog}>
                            <SelectTrigger>
                                <SelectValue placeholder="Choose Model" />
                            </SelectTrigger>
                            <SelectContent>
                                {tierModels.map((m) => (
                                    <SelectItem key={m.name} value={m.name}>
                                        {m.name} ({m.ram_gb}GB)
                                    </SelectItem>
                                ))}
                            </SelectContent>
                        </Select>
                    </div>

                    {/* Action Button */}
                    <div className="md:col-span-4 pb-0.5">
                        <Button
                            className="w-full bg-slate-900 hover:bg-slate-800"
                            onClick={handleSetModel}
                            disabled={switching || !selectedModel || selectedModel === currentModel}
                        >
                            {switching ? (
                                <>
                                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                                    Switching...
                                </>
                            ) : (
                                "Set Active Model"
                            )}
                        </Button>
                    </div>
                </div>

                {/* Info & RAM Visualization */}
                {activeModelObj && (
                    <div className="bg-slate-50 rounded-lg p-3 border border-slate-100">
                        <div className="flex items-center justify-between mb-2">
                            <span className="text-xs font-semibold uppercase text-slate-500">Estimated RAM Impact</span>
                            <span className={`text-xs font-bold px-2 py-0.5 rounded text-white ${getRamColor(activeModelObj.ram_gb)}`}>
                                {getRamText(activeModelObj.ram_gb)}
                            </span>
                        </div>
                        <div className="w-full bg-slate-200 rounded-full h-2.5 mb-2">
                            <div
                                className={`h-2.5 rounded-full ${getRamColor(activeModelObj.ram_gb)} transition-all duration-500`}
                                style={{ width: `${Math.min((activeModelObj.ram_gb / SYSTEM_RAM_GB) * 100, 100)}%` }}
                            ></div>
                        </div>
                        <p className="text-xs text-slate-600">
                            {activeModelObj.description}
                        </p>
                    </div>
                )}

                {/* Async Job Status */}
                {switching && jobStatus && (
                    <div className="bg-blue-50 border border-blue-200 rounded-lg p-3 flex items-start gap-3">
                        <Loader2 className="w-5 h-5 text-blue-600 animate-spin mt-0.5" />
                        <div>
                            <h4 className="text-sm font-semibold text-blue-900 capitalize">
                                Status: {jobStatus.status === "pending" ? "Queued" : jobStatus.status}
                            </h4>
                            <p className="text-xs text-blue-700 mt-1">
                                {jobStatus.details}
                            </p>
                        </div>
                    </div>
                )}
            </CardContent>
        </Card>
    );
};

export default SystemConfig;
