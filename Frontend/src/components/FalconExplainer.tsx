import { motion } from 'framer-motion';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { 
  Rocket, 
  Zap, 
  Database, 
  Settings, 
  Eye, 
  Cpu, 
  FileCode,
  Car,
  Bot,
  Satellite,
  Shield,
  TrendingUp,
  CheckCircle,
  ArrowRight,
  Play,
  Download,
  Camera,
  Code,
  Layers,
  RotateCcw,
  type LucideIcon
} from 'lucide-react';

const capabilities: Array<{
  icon: LucideIcon;
  title: string;
  description: string;
  color: string;
}> = [
  {
    icon: Eye,
    title: "Generate Realism",
    description: "Photorealistic RGB, depth maps, segmentation, bounding boxes & more with pixel-perfect precision.",
    color: "text-primary"
  },
  {
    icon: RotateCcw,
    title: "Scale Without Limits", 
    description: "Create thousands of scenes — from warehouses to spacewalks — instantly with infinite variations.",
    color: "text-accent"
  },
  {
    icon: Cpu,
    title: "Physics + Game Engine Powered",
    description: "Unreal Engine 5 + physics-based simulation + SDK/API control for unprecedented realism.",
    color: "text-primary"
  },
  {
    icon: FileCode,
    title: "Formats You Already Use",
    description: "Supports YOLO, COCO, Pascal VOC, custom JSON. Seamless integration with existing workflows.",
    color: "text-accent"
  }
];

const problems = [
  {
    problem: "Lack of rare-case data",
    solution: "Simulate anything: night, fog, fire, clutter",
    problemIcon: "📉",
    solutionIcon: "🧠"
  },
  {
    problem: "Human label errors",
    solution: "Pixel-perfect annotations",
    problemIcon: "🧼",
    solutionIcon: "✅"
  },
  {
    problem: "High real-world costs",
    solution: "Simulate once, reuse forever",
    problemIcon: "💸",
    solutionIcon: "🛰"
  },
  {
    problem: "Privacy concerns",
    solution: "No real cameras needed",
    problemIcon: "🕵",
    solutionIcon: "🚫"
  }
];

const workflowSteps: Array<{
  icon: LucideIcon;
  title: string;
  description: string;
}> = [
  { icon: Settings, title: "Define Environment", description: "Configure scene parameters" },
  { icon: Layers, title: "Add Objects", description: "Place and arrange objects" },
  { icon: Camera, title: "Configure Camera + Motion", description: "Set viewpoints and movement" },
  { icon: Database, title: "Choose Output Labels", description: "Select annotation formats" },
  { icon: Play, title: "Run Simulation", description: "Execute via SDK or GUI" },
  { icon: Download, title: "Download + Train", description: "Export and integrate data" }
];

const useCases: Array<{
  icon: LucideIcon;
  title: string;
  description: string;
  gradient: string;
}> = [
  {
    icon: Car,
    title: "Autonomous Vehicles",
    description: "Night driving, occlusion, crash simulation",
    gradient: "from-blue-500 to-cyan-500"
  },
  {
    icon: Bot,
    title: "Robotics",
    description: "Detect tools, conveyor workflows",
    gradient: "from-purple-500 to-pink-500"
  },
  {
    icon: Satellite,
    title: "Space & Defense",
    description: "EVA tools, satellite analysis",
    gradient: "from-primary to-accent"
  },
  {
    icon: Shield,
    title: "Disaster Response",
    description: "Simulate smoke, rubble, fire scenes",
    gradient: "from-orange-500 to-red-500"
  }
];

const impactStats = [
  { value: "+25%", label: "mAP with YOLOv8", sublabel: "on rare edge cases" },
  { value: "0%", label: "Label Error", sublabel: "pixel-perfect generation" },
  { value: "10X", label: "Faster Dataset", sublabel: "delivery time" },
  { value: "NASA", label: "Trusted by", sublabel: "DARPA, Nvidia alumni" }
];

export default function FalconExplainer() {
  return (
    <section className="py-20 px-4 relative overflow-hidden">
      {/* Animated Background */}
      <div className="absolute inset-0 bg-gradient-to-br from-background via-background/95 to-background/90">
        <div className="absolute inset-0 opacity-30">
          {Array.from({ length: 50 }).map((_, i) => (
            <motion.div
              key={i}
              className="absolute w-1 h-1 bg-primary/40 rounded-full"
              style={{
                left: `${Math.random() * 100}%`,
                top: `${Math.random() * 100}%`,
              }}
              animate={{
                opacity: [0.2, 1, 0.2],
                scale: [0.5, 1, 0.5],
              }}
              transition={{
                duration: 3 + Math.random() * 2,
                repeat: Infinity,
                delay: Math.random() * 2,
              }}
            />
          ))}
        </div>
      </div>

      <div className="container mx-auto relative z-10">
        {/* Hero Explainer Panel */}
        <motion.div
          className="text-center mb-16"
          initial={{ opacity: 0, y: 50 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          viewport={{ once: true }}
        >
          <Card className="glass-panel hover-tilt inline-block p-8 hover:cosmic-glow transition-all duration-500">
            <div className="flex items-center justify-center mb-4">
              <motion.div
                className="p-4 rounded-full bg-gradient-cosmic mr-4"
                whileHover={{ rotateY: 15, rotateX: 5 }}
                transition={{ duration: 0.3 }}
              >
                <Rocket className="w-8 h-8 text-primary-foreground" />
              </motion.div>
              <div className="text-left">
                <h2 className="text-4xl font-bold bg-gradient-to-r from-primary to-accent bg-clip-text text-transparent">
                  🚀 Falcon by Duality AI
                </h2>
                <p className="text-xl text-muted-foreground mt-2">
                  The Engine Behind the Mission
                </p>
              </div>
            </div>
            <p className="text-lg text-muted-foreground max-w-2xl">
              Simulation-grade synthetic data for next-gen AI training. Powering our orbital object detection through 
              physics-based digital twins and infinite scenario generation.
            </p>
          </Card>
        </motion.div>

        {/* Capabilities Grid */}
        <motion.div
          className="grid md:grid-cols-2 lg:grid-cols-4 gap-6 mb-16"
          initial={{ opacity: 0, y: 50 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.2 }}
          viewport={{ once: true }}
        >
          {capabilities.map((capability, index) => (
            <motion.div
              key={capability.title}
              initial={{ opacity: 0, y: 50 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: index * 0.1 }}
              viewport={{ once: true }}
            >
              <Card className="glass-panel hover-tilt hover:cosmic-glow transition-all duration-500 h-full">
                <div className="space-y-4">
                  <div className={`p-3 rounded-lg bg-gradient-cosmic w-fit ${capability.color}`}>
                    <capability.icon className="w-6 h-6" />
                  </div>
                  <h3 className="text-lg font-semibold">{capability.title}</h3>
                  <p className="text-sm text-muted-foreground leading-relaxed">
                    {capability.description}
                  </p>
                </div>
              </Card>
            </motion.div>
          ))}
        </motion.div>

        {/* Problems vs Solutions */}
        <motion.div
          className="mb-16"
          initial={{ opacity: 0, y: 50 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.4 }}
          viewport={{ once: true }}
        >
          <h3 className="text-3xl font-bold text-center mb-8 bg-gradient-to-r from-primary to-accent bg-clip-text text-transparent">
            Problems Falcon Solves
          </h3>
          <div className="grid md:grid-cols-2 gap-6">
            {problems.map((item, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, x: index % 2 === 0 ? -50 : 50 }}
                whileInView={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.6, delay: index * 0.1 }}
                viewport={{ once: true }}
              >
                <Card className="glass-panel hover:aurora-glow transition-all duration-500 group">
                  <div className="flex items-center justify-between">
                    <div className="flex-1">
                      <div className="flex items-center space-x-3 mb-3">
                        <span className="text-2xl">{item.problemIcon}</span>
                        <span className="text-sm text-muted-foreground">{item.problem}</span>
                      </div>
                      <ArrowRight className="w-4 h-4 text-primary mx-auto mb-3 group-hover:translate-x-1 transition-transform" />
                      <div className="flex items-center space-x-3">
                        <span className="text-2xl">{item.solutionIcon}</span>
                        <span className="text-sm font-medium text-primary">{item.solution}</span>
                      </div>
                    </div>
                  </div>
                </Card>
              </motion.div>
            ))}
          </div>
        </motion.div>

        {/* Developer Workflow Timeline */}
        <motion.div
          className="mb-16"
          initial={{ opacity: 0, y: 50 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.6 }}
          viewport={{ once: true }}
        >
          <h3 className="text-3xl font-bold text-center mb-8 bg-gradient-to-r from-accent to-primary bg-clip-text text-transparent">
            From Scene to Dataset
          </h3>
          <div className="relative">
            {/* Timeline line */}
            <div className="hidden lg:block absolute top-1/2 left-0 right-0 h-px bg-gradient-to-r from-primary via-accent to-primary opacity-30" />
            
            <div className="grid grid-cols-2 lg:grid-cols-6 gap-4">
              {workflowSteps.map((step, index) => (
                <motion.div
                  key={step.title}
                  className="relative"
                  initial={{ opacity: 0, y: 50 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.6, delay: index * 0.1 }}
                  viewport={{ once: true }}
                >
                  <Card className="glass-panel hover-tilt hover:cosmic-glow transition-all duration-500 text-center relative z-10">
                    <div className="space-y-3">
                      <div className="p-3 rounded-full bg-gradient-cosmic w-fit mx-auto">
                        <step.icon className="w-5 h-5 text-primary-foreground" />
                      </div>
                      <h4 className="font-semibold text-sm">{step.title}</h4>
                      <p className="text-xs text-muted-foreground">{step.description}</p>
                    </div>
                  </Card>
                  
                  {/* Step number */}
                  <div className="absolute -top-2 -right-2 w-6 h-6 rounded-full bg-primary text-primary-foreground text-xs flex items-center justify-center font-bold z-20">
                    {index + 1}
                  </div>
                </motion.div>
              ))}
            </div>
          </div>
        </motion.div>

        {/* Use Case Showcase */}
        <motion.div
          className="mb-16"
          initial={{ opacity: 0, y: 50 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.8 }}
          viewport={{ once: true }}
        >
          <h3 className="text-3xl font-bold text-center mb-8 bg-gradient-to-r from-primary to-accent bg-clip-text text-transparent">
            Industry Applications
          </h3>
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
            {useCases.map((useCase, index) => (
              <motion.div
                key={useCase.title}
                initial={{ opacity: 0, scale: 0.9 }}
                whileInView={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.6, delay: index * 0.1 }}
                viewport={{ once: true }}
              >
                <Card className="glass-panel hover-tilt hover:cosmic-glow transition-all duration-500 group overflow-hidden">
                  <div className="space-y-4">
                    <div className={`p-4 rounded-lg bg-gradient-to-br ${useCase.gradient} w-fit group-hover:scale-110 transition-transform duration-300`}>
                      <useCase.icon className="w-6 h-6 text-white" />
                    </div>
                    <h4 className="font-semibold">{useCase.title}</h4>
                    <p className="text-sm text-muted-foreground">{useCase.description}</p>
                  </div>
                </Card>
              </motion.div>
            ))}
          </div>
        </motion.div>

        {/* Impact Stats */}
        <motion.div
          className="mb-16"
          initial={{ opacity: 0, y: 50 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 1.0 }}
          viewport={{ once: true }}
        >
          <h3 className="text-3xl font-bold text-center mb-8 bg-gradient-to-r from-accent to-primary bg-clip-text text-transparent">
            Measurable Impact
          </h3>
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
            {impactStats.map((stat, index) => (
              <motion.div
                key={stat.label}
                initial={{ opacity: 0, y: 50 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: index * 0.1 }}
                viewport={{ once: true }}
              >
                <Card className="glass-panel hover:aurora-glow transition-all duration-500 text-center">
                  <div className="space-y-2">
                    <div className="text-3xl font-bold bg-gradient-to-r from-primary to-accent bg-clip-text text-transparent">
                      {stat.value}
                    </div>
                    <div className="font-semibold">{stat.label}</div>
                    <div className="text-xs text-muted-foreground">{stat.sublabel}</div>
                  </div>
                </Card>
              </motion.div>
            ))}
          </div>
        </motion.div>

        {/* Tech Footer & CTA */}
        <motion.div
          initial={{ opacity: 0, y: 50 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 1.2 }}
          viewport={{ once: true }}
        >
          <Card className="glass-panel hover:cosmic-glow transition-all duration-500">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 lg:gap-8 items-center">
              <div className="space-y-4 order-2 lg:order-1">
                <h4 className="text-xl sm:text-2xl font-semibold bg-gradient-to-r from-primary to-accent bg-clip-text text-transparent">
                  Enterprise-Grade Technology
                </h4>
                <div className="space-y-3 text-sm sm:text-base text-muted-foreground">
                  <div className="flex items-start space-x-3">
                    <CheckCircle className="w-4 h-4 sm:w-5 sm:h-5 text-primary mt-0.5 flex-shrink-0" />
                    <span>Powered by: Unreal Engine 5, physics-based rendering</span>
                  </div>
                  <div className="flex items-start space-x-3">
                    <CheckCircle className="w-4 h-4 sm:w-5 sm:h-5 text-primary mt-0.5 flex-shrink-0" />
                    <span>Integrates with: YOLOv8, Detectron2, Python SDK</span>
                  </div>
                  <div className="flex items-start space-x-3">
                    <CheckCircle className="w-4 h-4 sm:w-5 sm:h-5 text-primary mt-0.5 flex-shrink-0" />
                    <span>Available via API or GUI — currently in early access</span>
                  </div>
                </div>
              </div>
              
              <div className="text-center order-1 lg:order-2 lg:text-right space-y-4">
                <p className="text-sm sm:text-base text-muted-foreground">
                  Ready to transform your AI training pipeline?
                </p>
                <Button 
                  size="lg" 
                  className="cosmic-glow group w-full sm:w-auto"
                  onClick={() => window.open('https://falcon.duality.ai', '_blank', 'noopener,noreferrer')}
                >
                  <Code className="w-4 h-4 mr-2 group-hover:rotate-12 transition-transform" />
                  🔗 Request Access on falcon.duality.ai
                  <ArrowRight className="w-4 h-4 ml-2 group-hover:translate-x-1 transition-transform" />
                </Button>
              </div>
            </div>
          </Card>
        </motion.div>
      </div>
    </section>
  );
}