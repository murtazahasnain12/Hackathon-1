import React from 'react';
import DocItem from '@theme-original/DocItem';
import BreadcrumbNav from '@site/src/components/BreadcrumbNav';

export default function DocItemWrapper(props) {
  return (
    <>
      <BreadcrumbNav />
      <DocItem {...props} />
    </>
  );
}