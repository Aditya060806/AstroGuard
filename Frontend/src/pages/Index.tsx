import { useEffect } from 'react';
import Navigation from '@/components/Navigation';
import HeroSection from '@/components/HeroSection';
import DetectionLab from '@/components/DetectionLab';
import MetricsDashboard from '@/components/MetricsDashboard';
import FailureCases from '@/components/FailureCases';
import FalconExplainer from '@/components/FalconExplainer';
import FalconSync from '@/components/FalconSync';
import MobilePreview from '@/components/MobilePreview';
import ReportsViewer from '@/components/ReportsViewer';
import { motion } from 'framer-motion';

const Index = () => {
  useEffect(() => {
    // Smooth scrolling for the entire page
    document.documentElement.style.scrollBehavior = 'smooth';
    
    return () => {
      document.documentElement.style.scrollBehavior = 'auto';
    };
  }, []);

  return (
    <div className="relative">
      <Navigation />
      
      {/* Hero Section */}
      <motion.main
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 1 }}
      >
        <HeroSection />
        
        {/* Detection Lab Section */}
        <section id="detection">
          <DetectionLab />
        </section>
        
        {/* Metrics Dashboard Section */}
        <section id="metrics">
          <MetricsDashboard />
        </section>
        
        {/* Failure Cases Section */}
        <section id="failures">
          <FailureCases />
        </section>
        
        {/* Falcon Explainer Section */}
        <section id="falcon-explainer">
          <FalconExplainer />
        </section>
        
        {/* Falcon Sync Section */}
        <section id="falcon">
          <FalconSync />
        </section>
        
        {/* Mobile Preview Section */}
        <section id="mobile">
          <MobilePreview />
        </section>
        
        {/* Reports Section */}
        <section id="reports">
          <ReportsViewer />
        </section>
        
        {/* Footer */}
        <motion.footer
          className="py-20 px-4 border-t border-border/50"
          initial={{ opacity: 0, y: 50 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          viewport={{ once: true }}
        >
          <div className="container mx-auto text-center">
            <div className="space-y-4">
              <h3 className="text-2xl font-bold bg-gradient-to-r from-primary to-accent bg-clip-text text-transparent">
                AstroGuard: Orbital Sight
              </h3>
              <p className="text-muted-foreground max-w-2xl mx-auto">
                Advancing space exploration through intelligent object detection. 
                Powered by Falcon's synthetic digital twin technology for unparalleled accuracy in zero gravity environments.
              </p>
              <div className="flex items-center justify-center space-x-6 text-sm text-muted-foreground">
                <span>© 2024 AstroGuard Systems</span>
                <span>•</span>
                <span>Mission Critical AI</span>
                <span>•</span>
                <span>Orbital Safety Solutions</span>
              </div>
            </div>
          </div>
        </motion.footer>
      </motion.main>
    </div>
  );
};

export default Index;
