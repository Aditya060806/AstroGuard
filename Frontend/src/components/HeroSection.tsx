import { motion } from 'framer-motion';
import { Button } from '@/components/ui/button';
import { Satellite, Zap, Target, Database } from 'lucide-react';
import SpaceStation3D from './SpaceStation3D';

export default function HeroSection() {
  const scrollToSection = (sectionId: string) => {
    const element = document.getElementById(sectionId);
    if (element) {
      const offset = 80;
      const targetPosition = element.offsetTop - offset;
      window.scrollTo({
        top: targetPosition,
        behavior: 'smooth'
      });
    }
  };
  return (
    <section className="relative min-h-screen flex items-center justify-center py-12 sm:py-20 px-4 overflow-hidden">
      {/* Animated background elements */}
      <div className="absolute inset-0 overflow-hidden">
        <motion.div
          className="absolute top-20 left-10 w-2 h-2 bg-primary rounded-full sm:w-3 sm:h-3"
          animate={{
            scale: [1, 1.5, 1],
            opacity: [0.3, 1, 0.3],
          }}
          transition={{
            duration: 3,
            repeat: Infinity,
            ease: "easeInOut",
          }}
        />
        <motion.div
          className="absolute top-40 right-20 w-1 h-1 bg-accent rounded-full sm:w-2 sm:h-2"
          animate={{
            scale: [1, 2, 1],
            opacity: [0.2, 0.8, 0.2],
          }}
          transition={{
            duration: 4,
            repeat: Infinity,
            ease: "easeInOut",
            delay: 1,
          }}
        />
        <motion.div
          className="absolute bottom-32 left-1/4 w-3 h-3 bg-primary/50 rounded-full sm:w-4 sm:h-4"
          animate={{
            y: [-20, 20, -20],
            opacity: [0.2, 0.6, 0.2],
          }}
          transition={{
            duration: 5,
            repeat: Infinity,
            ease: "easeInOut",
            delay: 2,
          }}
        />
      </div>

      <div className="container mx-auto grid lg:grid-cols-2 gap-8 lg:gap-12 items-center relative z-10">
        {/* Left Content */}
        <motion.div
          className="space-y-6 sm:space-y-8 text-center lg:text-left"
          initial={{ opacity: 0, x: -50 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.8, ease: "easeOut" }}
        >
          <motion.div
            className="space-y-3 sm:space-y-4"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.2 }}
          >
            <div className="flex items-center justify-center lg:justify-start space-x-2 text-primary">
              <Satellite className="w-5 h-5 sm:w-6 sm:h-6" />
              <span className="text-xs sm:text-sm font-medium tracking-wider uppercase">Orbital Detection System</span>
            </div>
            
            <h1 className="text-4xl sm:text-5xl md:text-6xl lg:text-7xl font-bold bg-gradient-to-r from-primary via-accent to-primary bg-clip-text text-transparent leading-tight">
              AstroGuard
            </h1>
            
            <h2 className="text-2xl sm:text-3xl md:text-4xl lg:text-5xl font-light text-foreground/90">
              Orbital Sight
            </h2>
            
            <p className="text-base sm:text-lg lg:text-xl text-muted-foreground leading-relaxed max-w-lg mx-auto lg:mx-0">
              AI-Powered Object Detection in Zero Gravity. Advanced neural networks trained on Falcon's synthetic digital twin data for precise space station monitoring.
            </p>
          </motion.div>

          <motion.div
            className="flex flex-col sm:flex-row items-center justify-center lg:justify-start gap-3 sm:gap-4"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.4 }}
          >
            <Button 
              size="lg" 
              className="cosmic-glow hover:aurora-glow transition-all duration-300 w-full sm:w-auto"
              onClick={() => scrollToSection('detection')}
            >
              <Target className="w-4 h-4 sm:w-5 sm:h-5 mr-2" />
              Try Detection
            </Button>
            <Button 
              variant="outline" 
              size="lg" 
              className="glass border-primary/30 hover:bg-primary/10 w-full sm:w-auto"
              onClick={() => scrollToSection('metrics')}
            >
              <Zap className="w-4 h-4 sm:w-5 sm:h-5 mr-2" />
              View Metrics
            </Button>
            <Button 
              variant="ghost" 
              size="lg" 
              className="hover:bg-accent/10 w-full sm:w-auto"
              onClick={() => scrollToSection('falcon')}
            >
              <Database className="w-4 h-4 sm:w-5 sm:h-5 mr-2" />
              Falcon Sync
            </Button>
          </motion.div>

          {/* Stats */}
          <motion.div
            className="grid grid-cols-3 gap-3 sm:gap-6 pt-6 sm:pt-8"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.6 }}
          >
            <div className="text-center glass-card hover:cosmic-glow transition-all duration-300">
              <div className="text-lg sm:text-2xl font-bold text-primary">94.7%</div>
              <div className="text-xs sm:text-sm text-muted-foreground">mAP@0.5</div>
            </div>
            <div className="text-center glass-card hover:aurora-glow transition-all duration-300">
              <div className="text-lg sm:text-2xl font-bold text-accent">23ms</div>
              <div className="text-xs sm:text-sm text-muted-foreground">Inference</div>
            </div>
            <div className="text-center glass-card hover:cosmic-glow transition-all duration-300">
              <div className="text-lg sm:text-2xl font-bold text-primary">99.2%</div>
              <div className="text-xs sm:text-sm text-muted-foreground">Precision</div>
            </div>
          </motion.div>
        </motion.div>

        {/* Right - 3D Space Station */}
        <motion.div
          className="relative order-first lg:order-last"
          initial={{ opacity: 0, x: 50 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.8, delay: 0.3 }}
        >
          <div className="glass-panel hover-tilt h-64 sm:h-80 lg:h-96 flex items-center justify-center relative overflow-hidden orbital-glow">
            <div className="absolute inset-0 bg-gradient-cosmic opacity-30" />
            <div className="absolute inset-0 bg-gradient-aurora opacity-20 animate-pulse" />
            <SpaceStation3D />
            
            {/* Floating labels */}
            <motion.div
              className="absolute top-2 sm:top-4 left-2 sm:left-4 glass rounded-lg px-2 sm:px-3 py-1 text-xs"
              animate={{ y: [0, -5, 0] }}
              transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
            >
              <span className="text-primary">Live Detection Active</span>
            </motion.div>
            
            <motion.div
              className="absolute bottom-2 sm:bottom-4 right-2 sm:right-4 glass rounded-lg px-2 sm:px-3 py-1 text-xs"
              animate={{ y: [0, 5, 0] }}
              transition={{ duration: 2.5, repeat: Infinity, ease: "easeInOut", delay: 1 }}
            >
              <span className="text-accent">Orbital Sync: 100%</span>
            </motion.div>
          </div>
        </motion.div>
      </div>

      {/* Scroll indicator */}
      <motion.div
        className="absolute bottom-8 left-1/2 transform -translate-x-1/2"
        animate={{ y: [0, 10, 0] }}
        transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
      >
        <div className="w-6 h-10 border-2 border-primary/50 rounded-full flex justify-center">
          <motion.div
            className="w-1 h-3 bg-primary rounded-full mt-2"
            animate={{ y: [0, 16, 0] }}
            transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
          />
        </div>
      </motion.div>
    </section>
  );
}