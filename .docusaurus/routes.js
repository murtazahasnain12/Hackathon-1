import React from 'react';
import ComponentCreator from '@docusaurus/ComponentCreator';

export default [
  {
    path: '/physical-ai-book/__docusaurus/debug',
    component: ComponentCreator('/physical-ai-book/__docusaurus/debug', '12f'),
    exact: true
  },
  {
    path: '/physical-ai-book/__docusaurus/debug/config',
    component: ComponentCreator('/physical-ai-book/__docusaurus/debug/config', '4d3'),
    exact: true
  },
  {
    path: '/physical-ai-book/__docusaurus/debug/content',
    component: ComponentCreator('/physical-ai-book/__docusaurus/debug/content', 'a5b'),
    exact: true
  },
  {
    path: '/physical-ai-book/__docusaurus/debug/globalData',
    component: ComponentCreator('/physical-ai-book/__docusaurus/debug/globalData', 'abe'),
    exact: true
  },
  {
    path: '/physical-ai-book/__docusaurus/debug/metadata',
    component: ComponentCreator('/physical-ai-book/__docusaurus/debug/metadata', '587'),
    exact: true
  },
  {
    path: '/physical-ai-book/__docusaurus/debug/registry',
    component: ComponentCreator('/physical-ai-book/__docusaurus/debug/registry', '2ef'),
    exact: true
  },
  {
    path: '/physical-ai-book/__docusaurus/debug/routes',
    component: ComponentCreator('/physical-ai-book/__docusaurus/debug/routes', '1a5'),
    exact: true
  },
  {
    path: '/physical-ai-book/docs',
    component: ComponentCreator('/physical-ai-book/docs', 'c66'),
    routes: [
      {
        path: '/physical-ai-book/docs',
        component: ComponentCreator('/physical-ai-book/docs', '45c'),
        routes: [
          {
            path: '/physical-ai-book/docs',
            component: ComponentCreator('/physical-ai-book/docs', '037'),
            routes: [
              {
                path: '/physical-ai-book/docs',
                component: ComponentCreator('/physical-ai-book/docs', '0e9'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/physical-ai-book/docs/architecture/deployment',
                component: ComponentCreator('/physical-ai-book/docs/architecture/deployment', 'aea'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/physical-ai-book/docs/architecture/layered-approach',
                component: ComponentCreator('/physical-ai-book/docs/architecture/layered-approach', '2c4'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/physical-ai-book/docs/architecture/references',
                component: ComponentCreator('/physical-ai-book/docs/architecture/references', '070'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/physical-ai-book/docs/architecture/system-view',
                component: ComponentCreator('/physical-ai-book/docs/architecture/system-view', 'f58'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/physical-ai-book/docs/architecture/tool-mapping',
                component: ComponentCreator('/physical-ai-book/docs/architecture/tool-mapping', '892'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/physical-ai-book/docs/assets/code-examples/isaac-examples',
                component: ComponentCreator('/physical-ai-book/docs/assets/code-examples/isaac-examples', 'efb'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/physical-ai-book/docs/assets/code-examples/ros2-examples',
                component: ComponentCreator('/physical-ai-book/docs/assets/code-examples/ros2-examples', 'b80'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/physical-ai-book/docs/category/architecture',
                component: ComponentCreator('/physical-ai-book/docs/category/architecture', '5fb'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/physical-ai-book/docs/category/chapter-1-introduction-to-physical-ai--embodied-intelligence',
                component: ComponentCreator('/physical-ai-book/docs/category/chapter-1-introduction-to-physical-ai--embodied-intelligence', 'ed6'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/physical-ai-book/docs/category/chapter-2-ros-2-foundations-for-humanoids',
                component: ComponentCreator('/physical-ai-book/docs/category/chapter-2-ros-2-foundations-for-humanoids', '711'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/physical-ai-book/docs/category/chapter-3-digital-twins--physics-simulation',
                component: ComponentCreator('/physical-ai-book/docs/category/chapter-3-digital-twins--physics-simulation', '4e0'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/physical-ai-book/docs/category/chapter-4-ai-perception--learning-with-nvidia-isaac',
                component: ComponentCreator('/physical-ai-book/docs/category/chapter-4-ai-perception--learning-with-nvidia-isaac', '023'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/physical-ai-book/docs/category/chapter-5-vision-language-action-pipelines',
                component: ComponentCreator('/physical-ai-book/docs/category/chapter-5-vision-language-action-pipelines', 'c24'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/physical-ai-book/docs/category/chapter-6-autonomous-humanoid-capstone-architecture',
                component: ComponentCreator('/physical-ai-book/docs/category/chapter-6-autonomous-humanoid-capstone-architecture', '92f'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/physical-ai-book/docs/category/hardware--infrastructure',
                component: ComponentCreator('/physical-ai-book/docs/category/hardware--infrastructure', '06d'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/physical-ai-book/docs/chapter-1',
                component: ComponentCreator('/physical-ai-book/docs/chapter-1', 'ba0'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/physical-ai-book/docs/chapter-1/layered-architecture',
                component: ComponentCreator('/physical-ai-book/docs/chapter-1/layered-architecture', 'bc7'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/physical-ai-book/docs/chapter-1/physical-ai-intro',
                component: ComponentCreator('/physical-ai-book/docs/chapter-1/physical-ai-intro', 'bf3'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/physical-ai-book/docs/chapter-1/references',
                component: ComponentCreator('/physical-ai-book/docs/chapter-1/references', '3e7'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/physical-ai-book/docs/chapter-2',
                component: ComponentCreator('/physical-ai-book/docs/chapter-2', '466'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/physical-ai-book/docs/chapter-2/humanoid-ros',
                component: ComponentCreator('/physical-ai-book/docs/chapter-2/humanoid-ros', 'fad'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/physical-ai-book/docs/chapter-2/references',
                component: ComponentCreator('/physical-ai-book/docs/chapter-2/references', '591'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/physical-ai-book/docs/chapter-2/ros-foundations',
                component: ComponentCreator('/physical-ai-book/docs/chapter-2/ros-foundations', '44a'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/physical-ai-book/docs/chapter-3',
                component: ComponentCreator('/physical-ai-book/docs/chapter-3', '960'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/physical-ai-book/docs/chapter-3/digital-twins',
                component: ComponentCreator('/physical-ai-book/docs/chapter-3/digital-twins', '262'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/physical-ai-book/docs/chapter-3/physics-simulation',
                component: ComponentCreator('/physical-ai-book/docs/chapter-3/physics-simulation', 'fad'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/physical-ai-book/docs/chapter-3/references',
                component: ComponentCreator('/physical-ai-book/docs/chapter-3/references', 'fa6'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/physical-ai-book/docs/chapter-4',
                component: ComponentCreator('/physical-ai-book/docs/chapter-4', '458'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/physical-ai-book/docs/chapter-4/ai-perception',
                component: ComponentCreator('/physical-ai-book/docs/chapter-4/ai-perception', 'b91'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/physical-ai-book/docs/chapter-4/nvidia-isaac',
                component: ComponentCreator('/physical-ai-book/docs/chapter-4/nvidia-isaac', '5d8'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/physical-ai-book/docs/chapter-4/references',
                component: ComponentCreator('/physical-ai-book/docs/chapter-4/references', 'bea'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/physical-ai-book/docs/chapter-5',
                component: ComponentCreator('/physical-ai-book/docs/chapter-5', 'bd1'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/physical-ai-book/docs/chapter-5/llm-ros-actions',
                component: ComponentCreator('/physical-ai-book/docs/chapter-5/llm-ros-actions', 'e9e'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/physical-ai-book/docs/chapter-5/references',
                component: ComponentCreator('/physical-ai-book/docs/chapter-5/references', 'bff'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/physical-ai-book/docs/chapter-5/vision-language',
                component: ComponentCreator('/physical-ai-book/docs/chapter-5/vision-language', 'cfa'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/physical-ai-book/docs/chapter-5/whisper-integration',
                component: ComponentCreator('/physical-ai-book/docs/chapter-5/whisper-integration', 'af7'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/physical-ai-book/docs/chapter-6',
                component: ComponentCreator('/physical-ai-book/docs/chapter-6', '77f'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/physical-ai-book/docs/chapter-6/deployment-topology',
                component: ComponentCreator('/physical-ai-book/docs/chapter-6/deployment-topology', 'd0a'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/physical-ai-book/docs/chapter-6/references',
                component: ComponentCreator('/physical-ai-book/docs/chapter-6/references', 'b16'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/physical-ai-book/docs/chapter-6/system-integration',
                component: ComponentCreator('/physical-ai-book/docs/chapter-6/system-integration', '4c1'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/physical-ai-book/docs/chapter-6/workflow',
                component: ComponentCreator('/physical-ai-book/docs/chapter-6/workflow', '44c'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/physical-ai-book/docs/hardware/configurations',
                component: ComponentCreator('/physical-ai-book/docs/hardware/configurations', '738'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/physical-ai-book/docs/hardware/jetson-guide',
                component: ComponentCreator('/physical-ai-book/docs/hardware/jetson-guide', 'fa6'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/physical-ai-book/docs/hardware/lab-setup',
                component: ComponentCreator('/physical-ai-book/docs/hardware/lab-setup', '1e3'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/physical-ai-book/docs/hardware/references',
                component: ComponentCreator('/physical-ai-book/docs/hardware/references', '37c'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/physical-ai-book/docs/intro',
                component: ComponentCreator('/physical-ai-book/docs/intro', 'f2d'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/physical-ai-book/docs/navigation-checklist',
                component: ComponentCreator('/physical-ai-book/docs/navigation-checklist', '733'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/physical-ai-book/docs/summary-cross-reference',
                component: ComponentCreator('/physical-ai-book/docs/summary-cross-reference', 'b33'),
                exact: true,
                sidebar: "tutorialSidebar"
              }
            ]
          }
        ]
      }
    ]
  },
  {
    path: '*',
    component: ComponentCreator('*'),
  },
];
