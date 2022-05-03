import { createMedia } from "@artsy/fresnel";
import PropTypes from "prop-types";
import "semantic-ui-css/semantic.css";
import React, { Component } from "react";
import subsection_style from "./style.js";

import teaser from "./images/Teaser-SinNeRF.drawio.jpg";
import {
  Button,
  Container,
  Divider,
  Grid,
  Header,
  Icon,
  Image,
  List,
  Menu,
  Segment,
  Sidebar,
  Visibility,
} from "semantic-ui-react";

import MyCarousel from "./Carousel.js";

const Abstract = ({ mobile }) => (
  <div>
    <MyCarousel mobile={mobile} />

    <Image src={teaser} style={subsection_style}></Image>
    <p>
      <b>TL;DR:</b> Given only a single reference view as input, our novel
      semi-supervised framework trains a neural radiance field effectively. In
      contrast, previous method shows inconsistent geometry when synthesizing
      novel views.
    </p>
    <Header as="h2" textAlign="center" style={subsection_style}>
      Abstract
    </Header>
    <p>
      Despite the rapid development of Neural Radiance Field (NeRF), the
      necessity of dense covers largely prohibits its wider applications. While
      several recent works have attempted to address this issue, they either
      operate with sparse views (yet still, a few of them) or on simple
      objects/scenes. In this work, we consider a more ambitious task: training
      neural radiance field, over realistically complex visual scenes, by
      “looking only once”, i.e., using only a single view. To attain this goal,
      we present a <i>Single View NeRF</i> <b>(SinNeRF)</b> framework consisting
      of thoughtfully designed semantic and geometry regularizations.
      Specifically, SinNeRF constructs a semi-supervised learning process, where
      we introduce and propagate geometry pseudo labels and semantic pseudo
      labels to guide the progressive training process. Extensive experiments
      are conducted on complex scene benchmarks, including NeRF synthetic
      dataset, Local Light Field Fusion dataset and DTU dataset. We show that
      even without pre-training on multi-view datasets, SinNeRF can yield
      photo-realistic novel-view synthesis results. Under the single image
      setting, SinNeRF significantly outperforms the current state-of-the-art
      NeRF baselines in all cases.
    </p>
  </div>
);

export default Abstract;
