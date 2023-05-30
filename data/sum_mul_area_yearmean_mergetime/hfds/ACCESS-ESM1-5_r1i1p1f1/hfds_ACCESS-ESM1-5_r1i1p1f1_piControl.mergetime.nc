CDF  �   
      time       bnds      lon       lat          1   CDI       @Climate Data Interface version 2.2.1 (https://mpimet.mpg.de/cdi)   Conventions       CF-1.7 CMIP-6.2    source       oACCESS-ESM1.5 (2019): 
aerosol: CLASSIC (v1.0)
atmos: HadGAM2 (r1.1, N96; 192 x 145 longitude/latitude; 38 levels; top level 39255 m)
atmosChem: none
land: CABLE2.4
landIce: none
ocean: ACCESS-OM2 (MOM5, tripolar primarily 1deg; 360 x 300 longitude/latitude; 50 levels; top grid cell 0-10 m)
ocnBgchem: WOMBAT (same grid as ocean)
seaIce: CICE4.1 (same grid as ocean)    institution       aCommonwealth Scientific and Industrial Research Organisation, Aspendale, Victoria 3195, Australia      activity_id       CMIP   branch_method         standard   branch_time_in_child                 branch_time_in_parent         @�Հ       creation_date         2019-11-13T00:24:52Z   data_specs_version        01.00.30   
experiment        pre-industrial control     experiment_id         	piControl      external_variables        	areacello      forcing_index               	frequency         year   further_info_url      Phttps://furtherinfo.es-doc.org/CMIP6.CSIRO.ACCESS-ESM1-5.piControl.none.r1i1p1f1   grid      ,native atmosphere N96 grid (145x192 latxlon)   
grid_label        gn     history      VTue May 30 16:59:04 2023: cdo -O -s -a -f nc -mergetime /Users/benjamin/Data/p22c/CMIP6/sum_mul_area_yearmean/hfds/ACCESS-ESM1-5_r1i1p1f1/CMIP6.CMIP.CSIRO.ACCESS-ESM1-5.piControl.r1i1p1f1.Omon.hfds.gn.v20210316/hfds_Omon_ACCESS-ESM1-5_piControl_r1i1p1f1_gn_010101-060012.yearmean.mul.areacello_piControl_v20210316.fldsum.nc /Users/benjamin/Data/p22c/CMIP6/sum_mul_area_yearmean/hfds/ACCESS-ESM1-5_r1i1p1f1/CMIP6.CMIP.CSIRO.ACCESS-ESM1-5.piControl.r1i1p1f1.Omon.hfds.gn.v20210316/hfds_Omon_ACCESS-ESM1-5_piControl_r1i1p1f1_gn_060101-100012.yearmean.mul.areacello_piControl_v20210316.fldsum.nc /Users/benjamin/Data/p22c/CMIP6/sum_mul_area_yearmean/hfds/ACCESS-ESM1-5_r1i1p1f1/CMIP6.CMIP.CSIRO.ACCESS-ESM1-5.piControl.r1i1p1f1.Omon.hfds.gn.v20210316/hfds_Omon_ACCESS-ESM1-5_piControl_r1i1p1f1_gn_100101-110012.yearmean.mul.areacello_piControl_v20210316.fldsum.nc /Users/benjamin/github/d22a-mcdc/data/sum_mul_area_yearmean_mergetime/hfds/ACCESS-ESM1-5_r1i1p1f1/hfds_ACCESS-ESM1-5_r1i1p1f1_piControl.mergetime.nc
Thu Nov 03 22:05:11 2022: cdo -O -s -fldsum -setattribute,hfds@units=W m-2 m2 -mul -yearmean -selname,hfds /Users/benjamin/Data/p22b/CMIP6/hfds/ACCESS-ESM1-5_r1i1p1f1/CMIP6.CMIP.CSIRO.ACCESS-ESM1-5.piControl.r1i1p1f1.Omon.hfds.gn.v20210316/hfds_Omon_ACCESS-ESM1-5_piControl_r1i1p1f1_gn_010101-060012.nc /Users/benjamin/Data/p22b/CMIP6/areacello/ACCESS-ESM1-5_r1i1p1f1/CMIP6.CMIP.CSIRO.ACCESS-ESM1-5.piControl.r1i1p1f1.Ofx.areacello.gn.v20210316/areacello_Ofx_ACCESS-ESM1-5_piControl_r1i1p1f1_gn.nc /Users/benjamin/Data/p22c/CMIP6/sum_mul_area_yearmean/hfds/ACCESS-ESM1-5_r1i1p1f1/CMIP6.CMIP.CSIRO.ACCESS-ESM1-5.piControl.r1i1p1f1.Omon.hfds.gn.v20210316/hfds_Omon_ACCESS-ESM1-5_piControl_r1i1p1f1_gn_010101-060012.yearmean.mul.areacello_piControl_v20210316.fldsum.nc
2019-11-13T00:24:52Z ; CMOR rewrote data to be consistent with CMIP6, CF-1.7 CMIP-6.2 and CF standards.     initialization_index            institution_id        CSIRO      mip_era       CMIP6      nominal_resolution        250 km     notes         �Exp: ESM-piControl; Local ID: PI-01; Variable: hfds (['sfc_hflux_from_runoff', 'sfc_hflux_coupler', 'sfc_hflux_from_water_evap', 'sfc_hflux_from_water_prec', 'frazil_2d'])    parent_activity_id        CMIP   parent_experiment_id      piControl-spinup   parent_mip_era        CMIP6      parent_source_id      ACCESS-ESM1-5      parent_time_units         days since 0001-01-01      parent_variant_label      r1i1p1f1   physics_index               product       model-output   realization_index               realm         ocean      run_variant       jforcing: GHG, Oz, SA, Sl, Vl, BC, OC, (GHG = CO2, N2O, CH4, CFC11, CFC12, CFC113, HCFC22, HFC125, HFC134a)     	source_id         ACCESS-ESM1-5      source_type       AOGCM      sub_experiment        none   sub_experiment_id         none   table_id      Omon   
table_info        BCreation Date:(30 April 2019) MD5:e14f55f257cceafb2523e41244962371     title         'ACCESS-ESM1-5 output prepared for CMIP6    variable_id       hfds   variant_label         r1i1p1f1   version       	v20191112      cmor_version      3.4.0      tracking_id       1hdl:21.14100/8791656f-ee99-4e12-b36d-c1a0011e5606      license      $CMIP6 model data produced by CSIRO is licensed under a Creative Commons Attribution-ShareAlike 4.0 International License (https://creativecommons.org/licenses/).  Consult https://pcmdi.llnl.gov/CMIP6/TermsOfUse for terms of use governing CMIP6 output, including citation requirements and proper acknowledgment.  Further information about this data, including some limitations, can be found via the further_info_url (recorded as a global attribute in this file).  The data producers and data providers make no warranty, either express or implied, including, but not limited to, warranties of merchantability and fitness for a particular purpose. All liabilities arising from the supply of the information (including any liability arising in negligence) are excluded to the fullest extent permitted by law.   CDO       @Climate Data Operators version 2.2.0 (https://mpimet.mpg.de/cdo)         time                standard_name         time   	long_name         time   bounds        	time_bnds      units         day as %Y%m%d.%f   calendar      proleptic_gregorian    axis      T               �   	time_bnds                                 �   lon                standard_name         	longitude      	long_name         	longitude      units         degrees_east   axis      X               �   lat                standard_name         latitude   	long_name         latitude   units         degrees_north      axis      Y               �   hfds                   	   standard_name         'surface_downward_heat_flux_in_sea_water    	long_name         'Downward Heat Flux at Sea Water Surface    units         W m-2 m2   
_FillValue        `�x�   missing_value         `�x�   cell_methods      area: mean where sea time: mean    comment       {This is the net flux of heat entering the liquid water column through its upper surface (excluding any 'flux adjustment') .    cell_measures         area: areacello    history       o2019-11-13T00:24:49Z altered by CMOR: replaced missing value flag (-1e+20) with standard missing value (1e+20).             �1e+20).             �                A.��   A.�j    A/!�    V�!|A/&:�   A/!�    A/o�    VW`A/tZ�   A/o�    A/��    ��A/�z�   A/��    A0�    լB�A0M@   A0�    A0-    W$�3A0/]@   A0-    A0T    V��mA0Vm@   A0T    A0{%    �7P/A0}}@   A0{%    A0�5    ׄqA0��@   A0�5    A0�E    �9�1A0˝@   A0�E    A0�U    V#FA0�@   A0�U    A1e    ֮��A1�@   A1e    A1>u    ��A1@�@   A1>u    A1e�    �}�5A1g�@   A1e�    A1��    ֏�RA1��@   A1��    A1��    �L%A1��@   A1��    A1ڵ    ���A1�@   A1ڵ    A2�    ���A2@   A2�    A2(�    V�b�A2+-@   A2(�    A2O�    Up�SA2R=@   A2O�    A2v�    U�y�A2yM@   A2v�    A2�    U�b[A2�]@   A2�    A2�    W+	A2�m@   A2�    A2�%    ��]A2�}@   A2�%    A35    ��)A3�@   A35    A3:E    V�|�A3<�@   A3:E    A3aU    V)�6A3c�@   A3aU    A3�e    V;��A3��@   A3�e    A3�u    �ݙ�A3��@   A3�u    A3օ    U��SA3��@   A3օ    A3��    ֵ�A3��@   A3��    A4$�    ֠�A4&�@   A4$�    A4K�    �MA4N@   A4K�    A4r�    W��A4u@   A4r�    A4��    V��vA4�-@   A4��    A4��    ���sA4�=@   A4��    A4��    ���A4�M@   A4��    A5    ��ҁA5]@   A5    A56    U��A58m@   A56    A5]%    U��A5_}@   A5]%    A5�5    ��_sA5��@   A5�5    A5�E    ֞ZA5��@   A5�E    A5�U    �1�dA5ԭ@   A5�U    A5�e    �]�A5��@   A5�e    A6 u    U��iA6"�@   A6 u    A6G�    ԛ:�A6I�@   A6G�    A6n�    ֋�<A6p�@   A6n�    A6��    �[�%A6��@   A6��    A6��    U��A6�@   A6��    A6��    �9�TA6�@   A6��    A7
�    �!3A7-@   A7
�    A71�    V3�,A74=@   A71�    A7X�    ��~A7[M@   A7X�    A7�    �iOA7�]@   A7�    A7�    WaA7�m@   A7�    A7�%    ֠%�A7�}@   A7�%    A7�5    ��oKA7��@   A7�5    A8E    �I��A8�@   A8E    A8CU    V�]A8E�@   A8CU    A8je    Ա� A8l�@   A8je    A8�u    �>�A8��@   A8�u    A8��    �$BeA8��@   A8��    A8ߕ    �G�]A8��@   A8ߕ    A9�    W
%HA9�@   A9�    A9-�    V�T�A90@   A9-�    A9T�    �=��A9W@   A9T�    A9{�    �v�A9~-@   A9{�    A9��    T�ZA9�=@   A9��    A9��    ��LA9�M@   A9��    A9�    ���!A9�]@   A9�    A:    �3�KA:m@   A:    A:?%    ��;�A:A}@   A:?%    A:f5    Ճ$�A:h�@   A:f5    A:�E    דt�A:��@   A:�E    A:�U    �<yA:��@   A:�U    A:�e    V�Y�A:ݽ@   A:�e    A;u    ֝hA;�@   A;u    A;)�    W6�A;+�@   A;)�    A;P�    W~�BA;R�@   A;P�    A;w�    U��;A;y�@   A;w�    A;��    �fG�A;�@   A;��    A;��    ��CA;�@   A;��    A;��    WF{�A;�-@   A;��    A<�    W�A<=@   A<�    A<:�    U!&A<=M@   A<:�    A<b    V�'�A<d]@   A<b    A<�    ���A<�m@   A<�    A<�%    V���A<�}@   A<�%    A<�5    V�fyA<ٍ@   A<�5    A<�E    ��a�A= �@   A<�E    A=%U    ��;A='�@   A=%U    A=Le    U[%@A=N�@   A=Le    A=su    VnA=u�@   A=su    A=��    W$#�A=��@   A=��    A=��    ִ��A=��@   A=��    A=�    V��pA=��@   A=�    A>�    �K3�A>@   A>�    A>6�    �|�7A>9@   A>6�    A>]�    �"A>`-@   A>]�    A>��    ��<.A>�=@   A>��    A>��    V��CA>�M@   A>��    A>�    �f��A>�]@   A>�    A>�    V@A>�m@   A>�    A?!%    V��eA?#}@   A?!%    A?H5    V��A?J�@   A?H5    A?oE    �1��A?q�@   A?oE    A?�U    �>�A?��@   A?�U    A?�e    �'��A?��@   A?�e    A?�u    V+�A?��@   A?�u    A@   ��A@�   A@   A@J�   ���#A@v�   A@J�   A@,Ҁ   �Dr"A@-��   A@,Ҁ   A@@Z�   V@A@A��   A@@Z�   A@S�   ִ?�A@U�   A@S�   A@gj�   �$A@h��   A@gj�   A@z�   V���A@|�   A@z�   A@�z�   ք�A@���   A@�z�   A@��   V��A@�.�   A@��   A@���   U���A@���   A@���   A@��   �K�7A@�>�   A@��   A@ܚ�   U���A@�Ơ   A@ܚ�   A@�"�   Ug�qA@�N�   A@�"�   AA��   �<��AA֠   AA��   AA2�   �Z��AA^�   AA2�   AA*��   �#!�AA+�   AA*��   AA>B�   V��AA?n�   AA>B�   AAQʀ   V��AAR��   AAQʀ   AAeR�   �A��AAf~�   AAeR�   AAxڀ   WM�AAz�   AAxڀ   AA�b�   Vx-~AA���   AA�b�   AA��   ׃��AA��   AA��   AA�r�   VP}�AA���   AA�r�   AA���   ���AA�&�   AA���   AAڂ�   V��AAۮ�   AAڂ�   AA�
�   W(|cAA�6�   AA�
�   AB��   V�qAB��   AB��   AB�   �T"�ABF�   AB�   AB(��   W	��AB)Π   AB(��   AB<*�   �.AB=V�   AB<*�   ABO��   ��eABPޠ   ABO��   ABc:�   ծ;ABdf�   ABc:�   ABv   W/;:ABw�   ABv   AB�J�   �JAB�v�   AB�J�   AB�Ҁ   V?��AB���   AB�Ҁ   AB�Z�   � �AB���   AB�Z�   AB��   Ub,AB��   AB��   AB�j�   SBlGABٖ�   AB�j�   AB��   �y�AB��   AB��   AB�z�   �н�AC ��   AB�z�   AC�   �[I�AC.�   AC�   AC&��   ��qyAC'��   AC&��   AC:�   �"��AC;>�   AC:�   ACM��   �8�ZACNƠ   ACM��   ACa"�   �*�+ACbN�   ACa"�   ACt��   ׈5ACu֠   ACt��   AC�2�   V���AC�^�   AC�2�   AC���   �KmoAC��   AC���   AC�B�   V�5�AC�n�   AC�B�   AC�ʀ   U�B�AC���   AC�ʀ   AC�R�   ��}AC�~�   AC�R�   AC�ڀ   ո�AC��   AC�ڀ   AC�b�   V0�{AC���   AC�b�   AD�   VugAD�   AD�   AD$r�   ֝C�AD%��   AD$r�   AD7��   U��AD9&�   AD7��   ADK��   U֔NADL��   ADK��   AD_
�   V�w�AD`6�   AD_
�   ADr��   T�$ADs��   ADr��   AD��   ��ƫAD�F�   AD��   AD���   ӢӑAD�Π   AD���   AD�*�   ��`AD�V�   AD�*�   AD���   T���AD�ޠ   AD���   AD�:�   �EWgAD�f�   AD�:�   AD�   ֲ�cAD��   AD�   AD�J�   ��F�AD�v�   AD�J�   AEҀ   V 5�AE��   AEҀ   AE"Z�   U���AE#��   AE"Z�   AE5�   W��AE7�   AE5�   AEIj�   �1AEJ��   AEIj�   AE\�   V��AE^�   AE\�   AEpz�   W�sAEq��   AEpz�   AE��   �R�AE�.�   AE��   AE���   V���AE���   AE���   AE��   Uk��AE�>�   AE��   AE���   UMAE�Ơ   AE���   AE�"�   �q�AE�N�   AE�"�   AE媀   � @�AE�֠   AE媀   AE�2�   V�uAE�^�   AE�2�   AF��   V���AF�   AF��   AF B�   �۽�AF!n�   AF B�   AF3ʀ   �	�}AF4��   AF3ʀ   AFGR�   ֫K3AFH~�   AFGR�   AFZڀ   ��p�AF\�   AFZڀ   AFnb�   ��+AFo��   AFnb�   AF��   V�1�AF��   AF��   AF�r�   W �AF���   AF�r�   AF���   Ճ�lAF�&�   AF���   AF���   WüAF���   AF���   AF�
�   �4��AF�6�   AF�
�   AF㒀   �6FAF侠   AF㒀   AF��   ֪6AF�F�   AF��   AG
��   �r��AGΠ   AG
��   AG*�   ���AGV�   AG*�   AG1��   U	&!AG2ޠ   AG1��   AGE:�   WRw�AGFf�   AGE:�   AGX   �.5�AGY�   AGX   AGlJ�   �8��AGmv�   AGlJ�   AGҀ   WY}AG���   AGҀ   AG�Z�   �<9�AG���   AG�Z�   AG��   ֝�{AG��   AG��   AG�j�   �@��AG���   AG�j�   AG��   W/�}AG��   AG��   AG�z�   V��AG⦠   AG�z�   AG��   W�AG�.�   AG��   AH��   �#�uAH	��   AH��   AH�   ��}�AH>�   AH�   AH/��   �.�AH0Ơ   AH/��   AHC"�   ��>�AHDN�   AHC"�   AHV��   V�ɋAHW֠   AHV��   AHj2�   R�@�AHk^�   AHj2�   AH}��   ��D�AH~�   AH}��   AH�B�   ք��AH�n�   AH�B�   AH�ʀ   ��rAH���   AH�ʀ   AH�R�   UM��AH�~�   AH�R�   AH�ڀ   �"H�AH��   AH�ڀ   AH�b�   ��c�AH���   AH�b�   AH��   ���eAH��   AH��   AIr�   W	��AI��   AIr�   AI��   �I9�AI&�   AI��   AI-��   WM�AI.��   AI-��   AIA
�   ֐f�AIB6�   AIA
�   AIT��   ���AIU��   AIT��   AIh�   �'��AIiF�   AIh�   AI{��   �e��AI|Π   AI{��   AI�*�   U��AI�V�   AI�*�   AI���   W��AI�ޠ   AI���   AI�:�   �=T�AI�f�   AI�:�   AI�   �れAI��   AI�   AI�J�   V� �AI�v�   AI�J�   AI�Ҁ   �՘6AI���   AI�Ҁ   AJZ�   V���AJ��   AJZ�   AJ�   ף�WAJ�   AJ�   AJ+j�   �!;AJ,��   AJ+j�   AJ>�   ֠��AJ@�   AJ>�   AJRz�   ֐�AJS��   AJRz�   AJf�   W@'tAJg.�   AJf�   AJy��   �1��AJz��   AJy��   AJ��   V�BAJ�>�   AJ��   AJ���   W)��AJ�Ơ   AJ���   AJ�"�   ֆlQAJ�N�   AJ�"�   AJǪ�   W\DAJ�֠   AJǪ�   AJ�2�   ֨�fAJ�^�   AJ�2�   AJ   כ�AJ��   AJ   AKB�   V_̾AKn�   AKB�   AKʀ   �h7PAK��   AKʀ   AK)R�   �0߯AK*~�   AK)R�   AK<ڀ   U�t�AK>�   AK<ڀ   AKPb�   V�b1AKQ��   AKPb�   AKc�   �J��AKe�   AKc�   AKwr�   �d�AKx��   AKwr�   AK���   ����AK�&�   AK���   AK���   ֌�"AK���   AK���   AK�
�   �!�AK�6�   AK�
�   AKŒ�   V� �AKƾ�   AKŒ�   AK��   T���AK�F�   AK��   AK좀   V���AK�Π   AK좀   AL *�   � ��ALV�   AL *�   AL��   ����ALޠ   AL��   AL':�   �!%AL(f�   AL':�   AL:   ֣�*AL;�   AL:   ALNJ�   �@�\ALOv�   ALNJ�   ALaҀ   WBZ3ALb��   ALaҀ   ALuZ�   W��ALv��   ALuZ�   AL��   VL�FAL��   AL��   AL�j�   ֗ǸAL���   AL�j�   AL��   ����AL��   AL��   AL�z�   �)szALĦ�   AL�z�   AL��   V�UAL�.�   AL��   ALꊀ   ֏8AL붠   ALꊀ   AL��   W	�AL�>�   AL��   AM��   ֆ��AMƠ   AM��   AM%"�   �Ģ+AM&N�   AM%"�   AM8��   �~AM9֠   AM8��   AML2�   V�_AMM^�   AML2�   AM_��   U�|rAM`�   AM_��   AMsB�   �i\�AMtn�   AMsB�   AM�ʀ   V���AM���   AM�ʀ   AM�R�   U��AM�~�   AM�R�   AM�ڀ   V��AM��   AM�ڀ   AM�b�   ���AM�   AM�b�   AM��   ���AM��   AM��   AM�r�   U�-�AM鞠   AM�r�   AM���   ִ� AM�&�   AM���   AN��   �{�fAN��   AN��   AN#
�   V��YAN$6�   AN#
�   AN6��   ְ��AN7��   AN6��   ANJ�   �b�`ANKF�   ANJ�   AN]��   �K@�AN^Π   AN]��   ANq*�   ���ANrV�   ANq*�   AN���   U��AN�ޠ   AN���   AN�:�   ��:-AN�f�   AN�:�   AN�   �t�AN��   AN�   AN�J�   �\:�AN�v�   AN�J�   AN�Ҁ   W$�	AN���   AN�Ҁ   AN�Z�   �Sy�AN熠   AN�Z�   AN��   ׉��AN��   AN��   AOj�   W)�WAO��   AOj�   AO �   V��BAO"�   AO �   AO4z�   U�G�AO5��   AO4z�   AOH�   U�AOI.�   AOH�   AO[��   W�Z�AO\��   AO[��   AOo�   �W�AOp>�   AOo�   AO���   W{^AO�Ơ   AO���   AO�"�   �H�7AO�N�   AO�"�   AO���   ��rAO�֠   AO���   AO�2�   �AO�^�   AO�2�   AOк�   ��LAO��   AOк�   AO�B�   V�(�AO�n�   AO�B�   AO�ʀ   ׍�SAO���   AO�ʀ   AP�@   V���AP?P   AP�@   APm@   ֛^�APP   APm@   AP1@   U�xcAP�P   AP1@   AP"�@   ՝}�AP#�P   AP"�@   AP,�@   ��AP-OP   AP,�@   AP6}@   �Q~�AP7P   AP6}@   AP@A@   ��5AP@�P   AP@A@   APJ@   Vb:MAPJ�P   APJ@   APS�@   �AAPT_P   APS�@   AP]�@   Vj`�AP^#P   AP]�@   APgQ@   T��DAPg�P   APgQ@   APq@   ��>}APq�P   APq@   APz�@   V�9�AP{oP   APz�@   AP��@   W#AP�3P   AP��@   AP�a@   ׅ=bAP��P   AP�a@   AP�%@   ��AP��P   AP�%@   AP��@   Ԛ?@AP�P   AP��@   AP��@   ���AP�CP   AP��@   AP�q@   ֪{�AP�P   AP�q@   AP�5@   ��<gAP��P   AP�5@   AP��@   V�n�APɏP   AP��@   APҽ@   ���AP�SP   APҽ@   AP܁@   ��	9AP�P   AP܁@   AP�E@   ֵ[�AP��P   AP�E@   AP�	@   �1�XAP�P   AP�	@   AP��@   �H~AP�cP   AP��@   AQ�@   �¥hAQ'P   AQ�@   AQU@   V��AQ�P   AQU@   AQ@   U�4�AQ�P   AQ@   AQ �@   �e�AQ!sP   AQ �@   AQ*�@   V{�AQ+7P   AQ*�@   AQ4e@   ֢L�AQ4�P   AQ4e@   AQ>)@   �
xTAQ>�P   AQ>)@   AQG�@   �AQH�P   AQG�@   AQQ�@   V?�"AQRGP   AQQ�@   AQ[u@   ��*�AQ\P   AQ[u@   AQe9@   U���AQe�P   AQe9@   AQn�@   UV�6AQo�P   AQn�@   AQx�@   WQ0AQyWP   AQx�@   AQ��@   ��UAQ�P   AQ��@   AQ�I@   U�۟AQ��P   AQ�I@   AQ�@   V��AQ��P   AQ�@   AQ��@   ��kAQ�gP   AQ��@   AQ��@   ։�-AQ�+P   AQ��@   AQ�Y@   W��8AQ��P   AQ�Y@   AQ�@   ֗��AQ��P   AQ�@   AQ��@   U�WdAQ�wP   AQ��@   AQХ@   U�62AQ�;P   AQХ@   AQ�i@   ��YAQ��P   AQ�i@   AQ�-@   �}O8AQ��P   AQ�-@   AQ��@   ԙ��AQ�P   AQ��@   AQ��@   �=��AQ�KP   AQ��@   ARy@   �LnARP   ARy@   AR=@   ��%�AR�P   AR=@   AR@   ��AR�P   AR@   AR�@   V$��AR[P   AR�@   AR(�@   V��@AR)P   AR(�@   AR2M@   W��AR2�P   AR2M@   AR<@   �]�nAR<�P   AR<@   ARE�@   Tm�ARFkP   ARE�@   ARO�@   Um��ARP/P   ARO�@   ARY]@   �9��ARY�P   ARY]@   ARc!@   �Ov�ARc�P   ARc!@   ARl�@   Wu^�ARm{P   ARl�@   ARv�@   �M�`ARw?P   ARv�@   AR�m@   ����AR�P   AR�m@   AR�1@   ֙��AR��P   AR�1@   AR��@   Wk�AR��P   AR��@   AR��@   T�'�AR�OP   AR��@   AR�}@   �N9�AR�P   AR�}@   AR�A@   ֚�XAR��P   AR�A@   AR�@   W�AR��P   AR�@   AR��@   Wk��AR�_P   AR��@   AR΍@   Vg��AR�#P   AR΍@   AR�Q@   ��.xAR��P   AR�Q@   AR�@   R�e�AR�P   AR�@   AR��@   ��AR�oP   AR��@   AR��@   V"�AR�3P   AR��@   AR�a@   ��yAR��P   AR�a@   AS	%@   ջ�dAS	�P   AS	%@   AS�@   �&E+ASP   AS�@   AS�@   U�|�ASCP   AS�@   AS&q@   �Q�AS'P   AS&q@   AS05@   �bwAS0�P   AS05@   AS9�@   ���AS:�P   AS9�@   ASC�@   ���+ASDSP   ASC�@   ASM�@   � JASNP   ASM�@   ASWE@   ��NASW�P   ASWE@   ASa	@   V�JbASa�P   ASa	@   ASj�@   ��CASkcP   ASj�@   ASt�@   �h ASu'P   ASt�@   AS~U@   V�ӪAS~�P   AS~U@   AS�@   V�BFAS��P   AS�@   AS��@   V��AS�sP   AS��@   AS��@   U�AS�7P   AS��@   AS�e@   W~�AS��P   AS�e@   AS�)@   VE;AS��P   AS�)@   AS��@   ���AS��P   AS��@   AS±@   V�u�AS�GP   AS±@   AS�u@   U��AS�P   AS�u@   AS�9@   � ��AS��P   AS�9@   AS��@   VP��AS��P   AS��@   AS��@   V�7�AS�WP   AS��@   AS�@   ր�AS�P   AS�@   AS�I@   U�w�AS��P   AS�I@   AT@   W�OAT�P   AT@   AT�@   V��ATgP   AT�@   AT�@   ��� AT+P   AT�@   AT$Y@   UmŭAT$�P   AT$Y@   AT.@   W
OAT.�P   AT.@   AT7�@   V��	AT8wP   AT7�@   ATA�@   ����ATB;P   ATA�@   ATKi@   WV�ATK�P   ATKi@   ATU-@   ��;XATU�P   ATU-@   AT^�@   �JAT_�P   AT^�@   ATh�@   VU$ATiKP   ATh�@   ATry@   �Z �ATsP   ATry@   AT|=@   �nF
AT|�P   AT|=@   AT�@   �AT��P   AT�@   AT��@   ׂ�AT�[P   AT��@   AT��@   א�AT�P   AT��@   AT�M@   �̮AT��P   AT�M@   AT�@   V��&AT��P   AT�@   AT��@   ���AT�kP   AT��@   AT��@   ׏�DAT�/P   AT��@   AT�]@   �㷓AT��P   AT�]@   AT�!@   VٔLATԷP   AT�!@   AT��@   ���=AT�{P   AT��@   AT�@   W;%�AT�?P   AT�@   AT�m@   V�(�AT�P   AT�m@   AT�1@   �]"�AT��P   AT�1@   AU�@   WM;AU�P   AU�@   AU�@   W3��AUOP   AU�@   AU}@   լ�AUP   AU}@   AU"A@   W��AU"�P   AU"A@   AU,@   ֨�AU,�P   AU,@   AU5�@   ט��AU6_P   AU5�@   AU?�@   U&��AU@#P   AU?�@   AUIQ@   VA��AUI�P   AUIQ@   AUS@   �x��AUS�P   AUS@   AU\�@   �<�8AU]oP   AU\�@   AUf�@   �䕆AUg3P   AUf�@   AUpa@   ��#AUp�P   AUpa@   AUz%@   �q�AUz�P   AUz%@   AU��@   V��nAU�P   AU��@   AU��@   ֓# AU�CP   AU��@   AU�q@   ֗n�AU�P   AU�q@   AU�5@   U�#�AU��P   AU�5@   AU��@   �s�AU��P   AU��@   AU��@   ���AU�SP   AU��@   AU��@   WG?AU�P   AU��@   AU�E@   �y��AU��P   AU�E@   AU�	@   �ō'AUҟP   AU�	@   AU��@   ��L�AU�cP   AU��@   AU�@   W(SAU�'P   AU�@   AU�U@   U�#!AU��P   AU�U@   AU�@   �b�]AU��P   AU�@   AV�@   ����AVsP   AV�@   AV�@   V'�*AV7P   AV�@   AVe@   WG��AV�P   AVe@   AV )@   W@^AV �P   AV )@   AV)�@   �g��AV*�P   AV)�@   AV3�@   �Ҁ�AV4GP   AV3�@   AV=u@   ��AV>P   AV=u@   AVG9@   �j�4AVG�P   AVG9@   AVP�@   ��7AVQ�P   AVP�@   AVZ�@   Vs�AV[WP   AVZ�@   AVd�@   �7�ZAVeP   AVd�@   AVnI@   V�V4AVn�P   AVnI@   AVx@   ׀ۊAVx�P   AVx@   AV��@   �%��AV�gP   AV��@   AV��@   U�|}AV�+P   AV��@   AV�Y@   �47tAV��P   AV�Y@   AV�@   ֖�AV��P   AV�@   AV��@   �R�UAV�wP   AV��@   AV��@   ֬ŬAV�;P   AV��@   AV�i@   �E�&AV��P   AV�i@   AV�-@   W	��AV��P   AV�-@   AV��@   V� )AVЇP   AV��@   AVٵ@   V��AV�KP   AVٵ@   AV�y@   V���AV�P   AV�y@   AV�=@   WIeAV��P   AV�=@   AV�@   �P"�AV��P   AV�@   AW �@   ֧[\AW[P   AW �@   AW
�@   �"$�AWP   AW
�@   AWM@   ֊qoAW�P   AWM@   AW@   V{�[AW�P   AW@   AW'�@   U�AW(kP   AW'�@   AW1�@   V�7rAW2/P   AW1�@   AW;]@   V���AW;�P   AW;]@   AWE!@   ֣��AWE�P   AWE!@   AWN�@   ���AWO{P   AWN�@   AWX�@   �ǾwAWY?P   AWX�@   AWbm@   �%m7AWcP   AWbm@   AWl1@   W��cAWl�P   AWl1@   AWu�@   W�AWv�P   AWu�@   AW�@   �eE7AW�OP   AW�@   AW�}@   �7�7AW�P   AW�}@   AW�A@   �%��AW��P   AW�A@   AW�@   WRǓAW��P   AW�@   AW��@   W�KAW�_P   AW��@   AW��@   ՚�fAW�#P   AW��@   AW�Q@   ���VAW��P   AW�Q@   AW�@   V�Y}AWīP   AW�@   AW��@   ��-�AW�oP   AW��@   AWם@   ։U�AW�3P   AWם@   AW�a@   �$]�AW��P   AW�a@   AW�%@   W��AW�P   AW�%@   AW��@   T�YAW�P   AW��@   AW��@   Vu�2AW�CP   AW��@   AXq@   V��gAX	P   AXq@   AX5@   ֈ�AX�P   AX5@   AX�@   ՝�0AX�P   AX�@   AX%�@   ֻA0AX&SP   AX%�@   AX/�@   �"�AX0P   AX/�@   AX9E@   ��-�AX9�P   AX9E@   AXC	@   ��|�AXC�P   AXC	@   AXL�@   Wht�AXMcP   AXL�@   AXV�@   ֫��AXW'P   AXV�@   AX`U@   V�:AX`�P   AX`U@   AXj@   ՀirAXj�P   AXj@   AXs�@   �AXtsP   AXs�@   AX}�@   �mT�AX~7P   AX}�@   AX�e@   �/TjAX��P   AX�e@   AX�)@   V��AX��P   AX�)@   AX��@   �	� AX��P   AX��@   AX��@   WfJ
AX�GP   AX��@   AX�u@   ���AX�P   AX�u@   AX�9@   ֡��AX��P   AX�9@   AX��@   �-E�AXP   AX��@   AX��@   �HAX�WP   AX��@   AXՅ@   V�[�AX�P   AXՅ@   AX�I@   �K�AX��P   AX�I@   AX�@   օ�AX�P   AX�@   AX��@   �LqAX�gP   AX��@   AX��@   W7�AX�+P   AX��@   AYY@   Vĥ�AY�P   AYY@   AY@   VR%�AY�P   AY@   AY�@   �:J;AYwP   AY�@   AY#�@   �n,AY$;P   AY#�@   AY-i@   �*sAY-�P   AY-i@   AY7-@   Ur�?AY7�P   AY7-@   AY@�@   Պ<AYA�P   AY@�@   AYJ�@   �P�5AYKKP   AYJ�@   AYTy@   V���AYUP   AYTy@   AY^=@   ��AY^�P   AY^=@   AYh@   ֿ/AYh�P   AYh@   AYq�@   ��9AYr[P   AYq�@   AY{�@   Wi��AY|P   AY{�@   AY�M@   Uϵ�AY��P   AY�M@   AY�@   �<�QAY��P   AY�@   AY��@   U߄AY�kP   AY��@   AY��@   S�|�AY�/P   AY��@   AY�]@   ���AY��P   AY�]@   AY�!@   �$��AY��P   AY�!@   AY��@   Uw��AY�{P   AY��@   AYɩ@   W=�'AY�?P   AYɩ@   AY�m@   UD��AY�P   AY�m@   AY�1@   �qIHAY��P   AY�1@   AY��@   �5�AY�P   AY��@   AY�@   ֲL	AY�OP   AY�@   AY�}@   V��AY�P   AY�}@   AZA@   WV�AZ�P   AZA@   AZ@   ���AZ�P   AZ@   AZ�@   VpAZ_P   AZ�@   AZ!�@   ���0AZ"#P   AZ!�@   AZ+Q@   ֟lVAZ+�P   AZ+Q@   AZ5@   �X��AZ5�P   AZ5@   AZ>�@   VG͉AZ?oP   AZ>�@   AZH�@   V��RAZI3P   AZH�@   AZRa@   �,u�AZR�P   AZRa@   AZ\%@   �cA�AZ\�P   AZ\%@   AZe�@   V~�AZfP   AZe�@   AZo�@   �5�AZpCP   AZo�@   AZyq@   �'�<AZzP   AZyq@   AZ�5@   V�[�AZ��P   AZ�5@   AZ��@   ֢q�AZ��P   AZ��@   AZ��@   ��#�AZ�SP   AZ��@   AZ��@   T�Q�AZ�P   AZ��@   AZ�E@   V�N�AZ��P   AZ�E@   AZ�	@   V�ӟAZ��P   AZ�	@   AZ��@   W<OsAZ�cP   AZ��@   AZǑ@   Բ[�AZ�'P   AZǑ@   AZ�U@   �G��AZ��P   AZ�U@   AZ�@   ցj�AZۯP   AZ�@   AZ��@   �/�AZ�sP   AZ��@   AZ�@   ��kAZ�7P   AZ�@   AZ�e@   � �AZ��P   AZ�e@   A[)@   ֏�XA[�P   A[)@   A[�@   �9[�A[�P   A[�@   A[�@   VifVA[GP   A[�@   A[u@   T�e|A[ P   A[u@   A[)9@   ����A[)�P   A[)9@   A[2�@   V�
A[3�P   A[2�@   A[<�@   � Y7A[=WP   A[<�@   A[F�@   ��!A[GP   A[F�@   A[PI@   �Lk�A[P�P   A[PI@   A[Z@   �>��A[Z�P   A[Z@   A[c�@   U�JA[dgP   A[c�@   A[m�@   �7N�A[n+P   A[m�@   A[wY@   V A[w�P   A[wY@   A[�@   ��A[��P   A[�@   A[��@   V+aA[�wP   A[��@   A[��@   V���A[�;P   A[��@   A[�i@   ��b�A[��P   A[�i@   A[�-@   �<`A[��P   A[�-@   A[��@   �j,�A[��P   A[��@   A[��@   ֌��A[�KP   A[��@   A[�y@   �5�^A[�P   A[�y@   A[�=@   W^
oA[��P   A[�=@   A[�@   �ƑeA[ٗP   A[�@   A[��@   U0�A[�[P   A[��@   A[�@   ��dJA[�P   A[�@   A[�M@   T�,~A[��P   A[�M@   A\ @   �WcZA\ �P   A\ @   A\	�@   �"��A\
kP   A\	�@   A\�@   �d�tA\/P   A\�@   A\]@   �o�A\�P   A\]@   A\'!@   V�STA\'�P   A\'!@   A\0�@   V ��A\1{P   A\0�@   A\:�@   V��A\;?P   A\:�@   A\Dm@   ֢�OA\EP   A\Dm@   A\N1@   V\(-A\N�P   A\N1@   A\W�@   U�̡A\X�P   A\W�@   A\a�@   W*� A\bOP   A\a�@   A\k}@   W$׿A\lP   A\k}@   A\uA@   �N=�A\u�P   A\uA@   A\@   V���A\�P   A\@   A\��@   ��*A\�_P   A\��@   A\��@   �r"�A\�#P   A\��@   A\�Q@   ��o8A\��P   A\�Q@   A\�@   ծfQA\��P   A\�@   A\��@   V��A\�oP   A\��@   A\��@   W R`A\�3P   A\��@   A\�a@   ��kA\��P   A\�a@   A\�%@   �M�?A\ͻP   A\�%@   A\��@   �g��A\�P   A\��@   A\�@   �޳1A\�CP   A\�@   A\�q@   U��A\�P   A\�q@   A\�5@   �%Y�A\��P   A\�5@   A\��@   Ւj�A\��P   A\��@   A]�@   �O�A]SP   A]�@   A]�@   ��A]P   A]�@   A]E@   T��A]�P   A]E@   A]%	@   ֝�@A]%�P   A]%	@   A].�@   � FA]/cP   A].�@   A]8�@   U6�~A]9'P   A]8�@   A]BU@   U�M�A]B�P   A]BU@   A]L@   U�K9A]L�P   A]L@   A]U�@   Յ]A]VsP   A]U�@   A]_�@   V�fA]`7P   A]_�@   A]ie@   ��ݶA]i�P   A]ie@   A]s)@   �e�A]s�P   A]s)@   A]|�@   U�]�A]}�P   A]|�@   A]��@   V��A]�GP   A]��@   A]�u@   ԽA]�P   A]�u@   A]�9@   ׎��A]��P   A]�9@   A]��@   �|řA]��P   A]��@   A]��@   �/8�A]�WP   A]��@   A]��@   VIm�A]�P   A]��@   A]�I@   V}OA]��P   A]�I@   A]�@   ��0�A]ˣP   A]�@   A]��@   V�S2A]�gP   A]��@   A]ޕ@   V�(YA]�+P   A]ޕ@   A]�Y@   �Ǉ�A]��P   A]�Y@   A]�@   �;A]�P   A]�@   A]��@   W,��A]�wP   A]��@   A^�@   �x��A^;P   A^�@   A^i@   ֯�A^�P   A^i@   A^-@   Wlx(A^�P   A^-@   A^"�@   �/�AA^#�P   A^"�@   A^,�@   W	d�A^-KP   A^,�@   A^6y@   ՒF�A^7P   A^6y@   A^@=@   �WuA^@�P   A^@=@   A^J@   V�4A^J�P   A^J@   A^S�@   �-�}A^T[P   A^S�@   A^]�@   �gopA^^P   A^]�@   A^gM@   W�eA^g�P   A^gM@   A^q@   ַ�A^q�P   A^q@   A^z�@   WP��A^{kP   A^z�@   A^��@   ��qyA^�/P   A^��@   A^�]@   V�-�A^��P   A^�]@   A^�!@   ��}RA^��P   A^�!@   A^��@   �\�A^�{P   A^��@   A^��@   ֞�A^�?P   A^��@   A^�m@   ���A^�P   A^�m@   A^�1@   ּB�A^��P   A^�1@   A^��@   ֮�3A^ɋP   A^��@   A^ҹ@   ��A^�OP   A^ҹ@   A^�}@   WOA^�P   A^�}@   A^�A@   V��	A^��P   A^�A@   A^�@   U��%A^�P   A^�@   A^��@   ��xA^�_P   A^��@   A_�@   Vno�A_#P   A_�@   A_Q@   W���A_�P   A_Q@   A_@   �2�A_�P   A_@   A_ �@   T[��A_!oP   A_ �@   A_*�@   �1��A_+3P   A_*�@   A_4a@   �w�4A_4�P   A_4a@   A_>%@   U�%�A_>�P   A_>%@   A_G�@   �� A_HP   A_G�@   A_Q�@   �A�A_RCP   A_Q�@   A_[q@   ִxxA_\P   A_[q@   A_e5@   Up:A_e�P   A_e5@   A_n�@   � �A_o�P   A_n�@   A_x�@   V���A_ySP   A_x�@   A_��@   ֲ�(A_�P   A_��@   A_�E@   V���A_��P   A_�E@   A_�	@   �[��A_��P   A_�	@   A_��@   VIa�A_�cP   A_��@   A_��@   �q�A_�'P   A_��@   A_�U@   T��zA_��P   A_�U@   A_�@   V��^A_��P   A_�@   A_��@   V�.�A_�sP   A_��@   A_С@   ֜r�A_�7P   A_С@   A_�e@   T�]yA_��P   A_�e@   A_�)@   W"UA_�P   A_�)@   A_��@   W(��A_�P   A_��@   A_��@   ��X�A_�GP   A_��@   A` ��   V�wJA`�   A` ��   A`��   ����A`�   A`��   A`
~�   �	��A`
ɨ   A`
~�   A``�   �6}�A`��   A``�   A`B�   �8V�A`��   A`B�   A`$�   ��eA`o�   A`$�   A`�   ����A`Q�   A`�   A`"�   V���A`#3�   A`"�   A`'ʠ   ֠SA`(�   A`'ʠ   A`,��   ֻ,�A`,��   A`,��   A`1��   Uz!A`1٨   A`1��   A`6p�   ֿ^(A`6��   A`6p�   A`;R�   �~1�A`;��   A`;R�   A`@4�   V�M�A`@�   A`@4�   A`E�   Ԫz�A`Ea�   A`E�   A`I��   T���A`JC�   A`I��   A`Nڠ   �WRtA`O%�   A`Nڠ   A`S��   ���A`T�   A`S��   A`X��   ֗�*A`X�   A`X��   A`]��   V��A`]˨   A`]��   A`bb�   ֿ{A`b��   A`bb�   A`gD�   ֛�~A`g��   A`gD�   A`l&�   VHA`lq�   A`l&�   A`q�   V��A`qS�   A`q�   A`u�   Օ��A`v5�   A`u�   A`z̠   �ToeA`{�   A`z̠   A`��   ��A`��   A`��   A`���   V�n�A`�ۨ   A`���   A`�r�   WݚA`���   A`�r�   A`�T�   W��A`���   A`�T�   A`�6�   �S�A`���   A`�6�   A`��   ���A`�c�   A`��   A`���   U1�A`�E�   A`���   A`�ܠ   �O��A`�'�   A`�ܠ   A`���   �FOYA`�	�   A`���   A`���   ֛�A`��   A`���   A`���   ��^�A`�ͨ   A`���   A`�d�   V�[�A`���   A`�d�   A`�F�   ��g�A`���   A`�F�   A`�(�   W3��A`�s�   A`�(�   A`�
�   ��A`�U�   A`�
�   A`��   �@@?A`�7�   A`��   A`�Π   W�A`��   A`�Π   A`Ұ�   S7�2A`���   A`Ұ�   A`ג�   �iF)A`�ݨ   A`ג�   A`�t�   Vߒ�A`ܿ�   A`�t�   A`�V�   �[3A`ᡨ   A`�V�   A`�8�   �-AiA`惨   A`�8�   A`��   ֘��A`�e�   A`��   A`���   ��A`�G�   A`���   A`�ޠ   VX�uA`�)�   A`�ޠ   A`���   W��A`��   A`���   A`���   V��kA`���   A`���   Aa��   ֝.�AaϨ   Aa��   Aaf�   �+y�Aa��   Aaf�   AaH�   UIAa��   AaH�   Aa*�   �c�Aau�   Aa*�   Aa�   ��XAaW�   Aa�   Aa�   W��Aa9�   Aa�   Aa Р   VN8rAa!�   Aa Р   Aa%��   �bu�Aa%��   Aa%��   Aa*��   U���Aa*ߨ   Aa*��   Aa/v�   ����Aa/��   Aa/v�   Aa4X�   ��Aa4��   Aa4X�   Aa9:�   V��Aa9��   Aa9:�   Aa>�   ��
Aa>g�   Aa>�   AaB��   �269AaCI�   AaB��   AaG�   Uw��AaH+�   AaG�   AaL    Մw�AaM�   AaL    AaQ��   VϟMAaQ�   AaQ��   AaV��   ��	WAaVѨ   AaV��   Aa[h�   WC!Aa[��   Aa[h�   Aa`J�   �dNhAa`��   Aa`J�   Aae,�   ֑K�Aaew�   Aae,�   Aaj�   �~AajY�   Aaj�   Aan�   ׂ�%Aao;�   Aan�   AasҠ   Ul�XAat�   AasҠ   Aax��   W�_Aax��   Aax��   Aa}��   �0�Aa}�   Aa}��   Aa�x�   ��~Aa�è   Aa�x�   Aa�Z�   W8�Aa���   Aa�Z�   Aa�<�   �TA3Aa���   Aa�<�   Aa��   U��Aa�i�   Aa��   Aa� �   V�CAa�K�   Aa� �   Aa��   V�+�Aa�-�   Aa��   Aa�Ġ   V�X�Aa��   Aa�Ġ   Aa���   V:@ZAa��   Aa���   Aa���   S���Aa�Ө   Aa���   Aa�j�   ���yAa���   Aa�j�   Aa�L�   ����Aa���   Aa�L�   Aa�.�   �+�Aa�y�   Aa�.�   Aa��   �%��Aa�[�   Aa��   Aa��   V	GAa�=�   Aa��   Aa�Ԡ   �1�Aa��   Aa�Ԡ   Aa˶�   W*�Aa��   Aa˶�   AaИ�   �:�iAa��   AaИ�   Aa�z�   V�[Aa�Ũ   Aa�z�   Aa�\�   ����Aaڧ�   Aa�\�   Aa�>�   ����Aa߉�   Aa�>�   Aa� �   �C6ZAa�k�   Aa� �   Aa��   ��4OAa�M�   Aa��   Aa��   �?�=Aa�/�   Aa��   Aa�Ơ   ׋�WAa��   Aa�Ơ   Aa���   �z�OAa��   Aa���   Aa���   W%��Aa�ը   Aa���   Abl�   �:��Ab��   Abl�   AbN�   ���.Ab��   AbN�   Ab0�   V��8Ab{�   Ab0�   Ab�   ֤�\Ab]�   Ab�   Ab��   U�4Ab?�   Ab��   Ab֠   Ս[�Ab!�   Ab֠   Ab��   V��]Ab�   Ab��   Ab#��   �`�XAb#�   Ab#��   Ab(|�   W%��Ab(Ǩ   Ab(|�   Ab-^�   ր��Ab-��   Ab-^�   Ab2@�   ֧n�Ab2��   Ab2@�   Ab7"�   U�>�Ab7m�   Ab7"�   Ab<�   V#��Ab<O�   Ab<�   Ab@�   �Ae�AbA1�   Ab@�   AbEȠ   �s  AbF�   AbEȠ   AbJ��   VB�\AbJ��   AbJ��   AbO��   �oAbOר   AbO��   AbTn�   V_WAbT��   AbTn�   AbYP�   ����AbY��   AbYP�   Ab^2�   U�1Ab^}�   Ab^2�   Abc�   ���Abc_�   Abc�   Abg��   U��AbhA�   Abg��   Ablؠ   �`DAbm#�   Ablؠ   Abq��   ���Abr�   Abq��   Abv��   Ճ��Abv�   Abv��   Ab{~�   V�TAb{ɨ   Ab{~�   Ab�`�   �J?�Ab���   Ab�`�   Ab�B�   ֨اAb���   Ab�B�   Ab�$�   V��Ab�o�   Ab�$�   Ab��   V�˒Ab�Q�   Ab��   Ab��   V���Ab�3�   Ab��   Ab�ʠ   �9B�Ab��   Ab�ʠ   Ab���   ��wAb���   Ab���   Ab���   ֐okAb�٨   Ab���   Ab�p�   VL}Ab���   Ab�p�   Ab�R�   ��kAb���   Ab�R�   Ab�4�   V�SAb��   Ab�4�   Ab��   V�xXAb�a�   Ab��   Ab���   W6�Ab�C�   Ab���   Ab�ڠ   V�ΟAb�%�   Ab�ڠ   Abļ�   V9#Ab��   Abļ�   Abɞ�   �-��Ab��   Abɞ�   Ab΀�   աe�Ab�˨   Ab΀�   Ab�b�   ֯��Abӭ�   Ab�b�   Ab�D�   ��!�Ab؏�   Ab�D�   Ab�&�   ֩VoAb�q�   Ab�&�   Ab��   ֥Ab�S�   Ab��   Ab��   �2�Ab�5�   Ab��   Ab�̠   V�Ab��   Ab�̠   Ab�   �	��Ab���   Ab�   Ab���   V�0kAb�ۨ   Ab���   Ab�r�   W#"8Ab���   Ab�r�   Ab�T�   ��AAb���   Ab�T�   Ac6�   �@9Ac��   Ac6�   Ac	�   �s�Ac	c�   Ac	�   Ac��   �p��AcE�   Ac��   Acܠ   U[��Ac'�   Acܠ   Ac��   UU�Ac	�   Ac��   Ac��   �	�=Ac�   Ac��   Ac!��   V@)Ac!ͨ   Ac!��   Ac&d�   ���cAc&��   Ac&d�   Ac+F�   �N�Ac+��   Ac+F�   Ac0(�   ��Ac0s�   Ac0(�   Ac5
�   W@� Ac5U�   Ac5
�   Ac9�   U�.WAc:7�   Ac9�   Ac>Π   T[t:Ac?�   Ac>Π   AcC��   ֪�hAcC��   AcC��   AcH��   ֽ�YAcHݨ   AcH��   AcMt�   V�AcM��   AcMt�   AcRV�   W<ހAcR��   AcRV�   AcW8�   W�AcW��   AcW8�   Ac\�   �,��Ac\e�   Ac\�   Ac`��   ���AcaG�   Ac`��   Aceޠ   W��Acf)�   Aceޠ   Acj��   W��Ack�   Acj��   Aco��   ��Aco��   Aco��   Act��   ��QActϨ   Act��   Acyf�   ֓�5Acy��   Acyf�   Ac~H�   V_�EAc~��   Ac~H�   Ac�*�   ���Ac�u�   Ac�*�   Ac��   �.�Ac�W�   Ac��   Ac��   �v�Ac�9�   Ac��   Ac�Р   ՝��Ac��   Ac�Р   Ac���   �(7�Ac���   Ac���   Ac���   V :Ac�ߨ   Ac���   Ac�v�   ։�2Ac���   Ac�v�   Ac�X�   կ Ac���   Ac�X�   Ac�:�   փ[`Ac���   Ac�:�   Ac��   W  Ac�g�   Ac��   Ac���   U��HAc�I�   Ac���   Ac��   ���Ac�+�   Ac��   Ac�    ִAc��   Ac�    Ac¤�   V��Ac��   Ac¤�   Acǆ�   �-LAc�Ѩ   Acǆ�   Ac�h�   ���Ac̳�   Ac�h�   Ac�J�   ֖'�Acѕ�   Ac�J�   Ac�,�   �tiAc�w�   Ac�,�   Ac��   W��Ac�Y�   Ac��   Ac��   א��Ac�;�   Ac��   Ac�Ҡ   �;�Ac��   Ac�Ҡ   Ac鴠   V�O�Ac���   Ac鴠   Ac   �Մ_Ac��   Ac   Ac�x�   T��Ac�è   Ac�x�   Ac�Z�   V�W,Ac���   Ac�Z�   Ac�<�   �Q��Ac���   Ac�<�   Ad�   �ئ]Adi�   Ad�   Ad �   �ے*AdK�   Ad �   Ad�   �8e�Ad-�   Ad�   AdĠ   �݅
Ad�   AdĠ   Ad��   ���Ad�   Ad��   Ad��   T�-�AdӨ   Ad��   Adj�   Vw\�Ad��   Adj�   Ad$L�   Ԏ"�Ad$��   Ad$L�   Ad).�   �K�/Ad)y�   Ad).�   Ad.�   ֧kyAd.[�   Ad.�   Ad2�   U`<Ad3=�   Ad2�   Ad7Ԡ   ���qAd8�   Ad7Ԡ   Ad<��   V9MAd=�   Ad<��   AdA��   ֻ�AdA�   AdA��   AdFz�   ֙RgAdFŨ   AdFz�   AdK\�   WIF�AdK��   AdK\�   AdP>�   ՅwAdP��   AdP>�   AdU �   V/ۗAdUk�   AdU �   AdZ�   V�qAdZM�   AdZ�   Ad^�   U3�JAd_/�   Ad^�   AdcƠ   �Y�FAdd�   AdcƠ   Adh��   V�2�Adh�   Adh��   Adm��   V)��Admը   Adm��   Adrl�   �
��Adr��   Adrl�   AdwN�   U�<Adw��   AdwN�   Ad|0�   V��Ad|{�   Ad|0�   Ad��   ՂlAd�]�   Ad��   Ad���   V�$Ad�?�   Ad���   Ad�֠   Տ#jAd�!�   Ad�֠   Ad���   ���Ad��   Ad���   Ad���   ���Ad��   Ad���   Ad�|�   V�ÓAd�Ǩ   Ad�|�   Ad�^�   Uw�NAd���   Ad�^�   Ad�@�   �#��Ad���   Ad�@�   Ad�"�   �R��Ad�m�   Ad�"�   Ad��   V(/Ad�O�   Ad��   Ad��   UEPAAd�1�   Ad��   Ad�Ƞ   ֟u$Ad��   Ad�Ƞ   Ad���   ��geAd���   Ad���   Ad���   Ud	Ad�ר   Ad���   Ad�n�   ��Y�AdŹ�   Ad�n�   Ad�P�   �p��Adʛ�   Ad�P�   Ad�2�   �Ll!Ad�}�   Ad�2�   Ad��   ��k�Ad�_�   Ad��   Ad���   W��Ad�A�   Ad���   Ad�ؠ   �*�iAd�#�   Ad�ؠ   Ad⺠   U�3�Ad��   Ad⺠   Ad眠   ֆ�wAd��   Ad眠   Ad�~�   V3KdAd�ɨ   Ad�~�   Ad�`�   �;%�Ad�   Ad�`�   Ad�B�   U�ՕAd���   Ad�B�   Ad�$�   Vr�WAd�o�   Ad�$�   Ae �   ���