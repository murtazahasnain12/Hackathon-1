import React from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import { useLocation } from '@docusaurus/router';
import { useDoc } from '@docusaurus/theme-common/internal';
import styles from './styles.module.css';

// Define breadcrumb mappings for our Physical AI book
const BREADCRUMB_MAPPINGS = {
  '/docs/intro': [
    { label: 'Home', path: '/' },
    { label: 'Introduction', path: '/docs/intro' }
  ],
  '/docs/chapter-1': [
    { label: 'Home', path: '/' },
    { label: 'Chapter 1', path: '/docs/chapter-1' }
  ],
  '/docs/chapter-1/physical-ai-intro': [
    { label: 'Home', path: '/' },
    { label: 'Chapter 1', path: '/docs/chapter-1' },
    { label: 'Physical AI Fundamentals', path: '/docs/chapter-1/physical-ai-intro' }
  ],
  '/docs/chapter-1/layered-architecture': [
    { label: 'Home', path: '/' },
    { label: 'Chapter 1', path: '/docs/chapter-1' },
    { label: 'Layered Architecture', path: '/docs/chapter-1/layered-architecture' }
  ],
  '/docs/chapter-1/references': [
    { label: 'Home', path: '/' },
    { label: 'Chapter 1', path: '/docs/chapter-1' },
    { label: 'References', path: '/docs/chapter-1/references' }
  ],
  '/docs/chapter-2': [
    { label: 'Home', path: '/' },
    { label: 'Chapter 2', path: '/docs/chapter-2' }
  ],
  '/docs/chapter-2/ros-foundations': [
    { label: 'Home', path: '/' },
    { label: 'Chapter 2', path: '/docs/chapter-2' },
    { label: 'ROS 2 Fundamentals', path: '/docs/chapter-2/ros-foundations' }
  ],
  '/docs/chapter-2/humanoid-ros': [
    { label: 'Home', path: '/' },
    { label: 'Chapter 2', path: '/docs/chapter-2' },
    { label: 'Humanoid ROS', path: '/docs/chapter-2/humanoid-ros' }
  ],
  '/docs/chapter-2/references': [
    { label: 'Home', path: '/' },
    { label: 'Chapter 2', path: '/docs/chapter-2' },
    { label: 'References', path: '/docs/chapter-2/references' }
  ],
  '/docs/chapter-3': [
    { label: 'Home', path: '/' },
    { label: 'Chapter 3', path: '/docs/chapter-3' }
  ],
  '/docs/chapter-3/physics-simulation': [
    { label: 'Home', path: '/' },
    { label: 'Chapter 3', path: '/docs/chapter-3' },
    { label: 'Physics Simulation', path: '/docs/chapter-3/physics-simulation' }
  ],
  '/docs/chapter-3/digital-twins': [
    { label: 'Home', path: '/' },
    { label: 'Chapter 3', path: '/docs/chapter-3' },
    { label: 'Digital Twins', path: '/docs/chapter-3/digital-twins' }
  ],
  '/docs/chapter-3/references': [
    { label: 'Home', path: '/' },
    { label: 'Chapter 3', path: '/docs/chapter-3' },
    { label: 'References', path: '/docs/chapter-3/references' }
  ]
};

// Add more specific paths as needed
const generateBreadcrumbs = (pathname) => {
  // Check if we have a specific mapping
  if (BREADCRUMB_MAPPINGS[pathname]) {
    return BREADCRUMB_MAPPINGS[pathname];
  }

  // For other paths, create a simple breadcrumb based on path structure
  const pathParts = pathname.split('/').filter(part => part);
  const breadcrumbs = [{ label: 'Home', path: '/' }];

  let currentPath = '';
  pathParts.forEach((part, index) => {
    currentPath += '/' + part;
    if (index === pathParts.length - 1) {
      // Last part is the current page, so we don't make it a link
      breadcrumbs.push({ label: formatPathPart(part), path: currentPath, isCurrent: true });
    } else {
      breadcrumbs.push({ label: formatPathPart(part), path: currentPath });
    }
  });

  return breadcrumbs;
};

const formatPathPart = (part) => {
  // Convert kebab-case to Title Case
  return part
    .split('-')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
};

const BreadcrumbNav = () => {
  const location = useLocation();
  const breadcrumbs = generateBreadcrumbs(location.pathname);

  if (breadcrumbs.length <= 1) {
    return null; // Don't show breadcrumbs if there's only one item (home)
  }

  return (
    <nav className={clsx(styles.breadcrumbNav, 'breadcrumb-nav')} aria-label="Breadcrumb">
      <ol className={clsx(styles.breadcrumb, 'breadcrumb')}>
        {breadcrumbs.map((item, index) => (
          <li key={item.path} className={clsx(styles.breadcrumbItem, 'breadcrumb__item')}>
            {index === breadcrumbs.length - 1 ? (
              // Current page - not a link
              <span className={clsx(styles.breadcrumbCurrent, 'breadcrumb__link--active')}>
                {item.label}
              </span>
            ) : (
              // Regular breadcrumb item - is a link
              <Link
                to={item.path}
                className={clsx(styles.breadcrumbLink, 'breadcrumb__link')}
              >
                {item.label}
              </Link>
            )}
            {index < breadcrumbs.length - 1 && (
              <span className={clsx(styles.breadcrumbSeparator, 'breadcrumb__label')} aria-hidden="true">
                {' > '}
              </span>
            )}
          </li>
        ))}
      </ol>
    </nav>
  );
};

export default BreadcrumbNav;